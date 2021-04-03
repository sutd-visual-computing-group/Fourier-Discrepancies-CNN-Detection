# General modules
import os, math, argparse, json, glob
import warnings
warnings.filterwarnings("ignore")

# Imaging/ scientific modules
import numpy as np
from PIL import Image 

# Pytorch modules
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# Webdataset module (https://github.com/tmbdev/webdataset)
import webdataset as wds

# Lightning module
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, GPUStatsMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers

# Other modules
from tqdm import tqdm

# Importing our own modules here
from settings import NUM_TRAINING_FILES
from dataset import CelebADataModule
from module import GAN



def main():
    # > Setup command line arguments
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument('--dataset', default='celeba', choices=['celeba', 'lsun'])
    parser.add_argument('--url', default='datasets/celeba/celeba-{000000..000007}.tar')
    parser.add_argument('--img_size', default=128, type=int)
    parser.add_argument('--n_upsamplings', type=int, default=5)

    # GAN arguments
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--adversarial_loss_mode', default='wgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
    parser.add_argument('--gradient_penalty_mode', default='0-gp', choices=['none', '1-gp', '0-gp', 'lp'])
    parser.add_argument('--gradient_penalty_sample_mode', default='line', choices=['line', 'real', 'fake', 'dragan'])
    parser.add_argument('--gradient_penalty_weight', type=float, default=10.0)
    parser.add_argument('--gradient_penalty_d_norm', default='layer_norm', choices=['instance_norm', 'layer_norm'])
    parser.add_argument('--n_d', type=int, default=5)  # # d updates per g update


    # Metrics arguments
    parser.add_argument('--fid_measure_samples', type=int, default=10000)
    parser.add_argument('--fid_ref_data_dir', default='datasets/celeba/fid_eval_10k/')
    parser.add_argument('--inference_batch_size', type=int, default=2000)


    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--b1', type=float, default=0.5)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--output_dir', default='output/')
    
    # Other details
    parser.add_argument('--setup_name', default='BASELINE')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True)
    
    # Distributed learning arguments
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.adversarial_loss_mode), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.adversarial_loss_mode, args.setup_name), exist_ok=True)
    save_dir = os.path.join(args.output_dir, args.adversarial_loss_mode, args.setup_name)
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Resume training if checkpoints already exist
    if os.path.exists(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0 :
        ckpt_files = glob.glob(ckpt_dir+'/*')
        latest_ckpt = max(ckpt_files, key=os.path.getctime)
        resume_from_checkpoint = latest_ckpt
    else:
        resume_from_checkpoint = None
    
    # Seed for research reprodicibility
    pl.seed_everything(args.random_seed)

    # Create image transforms
    image_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset
    if args.dataset == 'celeba' or args.dataset == 'lsun':
        data_module = CelebADataModule( args.url, args.img_size, image_transform, NUM_TRAINING_FILES[args.dataset],
                                        args.batch_size, args.num_workers )

    # Create GAN model
    model = GAN(latent_dim = args.z_dim,
                 dim = args.dim,
                 n_upsamplings = args.n_upsamplings,
                 lr = args.lr,
                 b1 = args.b1,
                 b2 = args.b2,
                 batch_size = args.batch_size,
                 setup_name=args.setup_name,
                 adversarial_mode = args.adversarial_loss_mode,
                 gradient_penalty_mode = args.gradient_penalty_mode,
                 gradient_penalty_sample_mode = args.gradient_penalty_sample_mode,
                 gradient_penalty_weight = args.gradient_penalty_weight,
                 gradient_penalty_d_norm = args.gradient_penalty_d_norm,
                 n_d = args.n_d,
                 sample_every = args.sample_every,
                 expected_epochs = args.epochs,
                 expected_dataset = args.dataset,
                 fid_measure_samples = args.fid_measure_samples,
                 fid_ref_data_dir = args.fid_ref_data_dir,
                 inference_batch_size = args.inference_batch_size,
                )

    # Create callbacks
    callbacks = [   LearningRateMonitor(logging_interval="epoch"), 
                    GPUStatsMonitor(),
                    ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch}-{fid10k:.2f}', save_top_k=5, monitor='fid10k'),
                    EarlyStopping(monitor='fid10k', min_delta=0.00, patience=5, verbose=True, mode='min'),
                ]

    tb_logger = pl_loggers.TensorBoardLogger(args.output_dir, 
                                            name = args.adversarial_loss_mode, 
                                            version = args.setup_name )

    # Create lightning training module
    accelerator = None if args.gpus == 1 else 'dp'
    trainer = pl.Trainer(accelerator=accelerator,
                        gpus = args.gpus, 
                        max_epochs = args.epochs, 
                        progress_bar_refresh_rate = 20, 
                        deterministic=True,
                        default_root_dir=save_dir, 
                        weights_save_path=save_dir, 
                        callbacks=callbacks,
                        resume_from_checkpoint=resume_from_checkpoint,
                        benchmark=True,
                        logger=tb_logger)

    # Save settings
    with open(os.path.join(trainer.log_dir, 'config.txt'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    # Train
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()