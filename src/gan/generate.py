# General modules
import os, math, argparse, json, glob

# Imaging/ scientific modules
import numpy as np
import PIL

# Pytorch modules
import torch

# Lightning module
import pytorch_lightning as pl

# Other modules
from tqdm import tqdm

# Importing our own modules here
from module import ConvGenerator


def generate_samples(G, 
                    latent_dim, 
                    batch_size, 
                    n_samples, 
                    save_dir,
                    sample_dir_name):
    
    # Use cuda if available
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Load generator weights from checkpoint
    G.to(device)

    # Set G to eval mode
    G.eval()

    # Convert samples into numpy files and save images as PNG with max quality
    gen_samples_dir = os.path.join(save_dir, sample_dir_name)
 
    if not os.path.exists(gen_samples_dir):
        os.mkdir(gen_samples_dir)

    for i in range( int(math.ceil(n_samples/batch_size)) ):
        # Fixed noise for sampling
        z = torch.randn( min( batch_size, int(n_samples-batch_size*i) ), latent_dim, 1, 1).to(device)

        # Generate samples
        with torch.no_grad():
            samples = G(z)

        save_samples_as_png(samples, gen_samples_dir, i*batch_size)


def save_samples_as_png(samples_tensor, save_loc, start_index):
    #samples_tensor = samples_tensor.cpu().numpy()
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]

    for i in tqdm(range(samples_tensor.shape[0])):
        index = i + start_index
        img = (samples_tensor[i, :, :, :].permute(1, 2, 0).cpu().numpy()*std) + mean
        img_pil = PIL.Image.fromarray(np.uint8(img*255.0))
        img_pil.save(os.path.join(save_loc, "{}.png".format(index)), quality=95, subsampling=-1)

    return


def main():
    # > Setup command line arguments
    parser = argparse.ArgumentParser()

    # GAN arguments
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--n_upsamplings', type=int, default=5)
    parser.add_argument('--adversarial_loss_mode', default='gan', choices=['gan', 'lsgan', 'wgan'])

    # Directories arguments
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--setup_name', default='BASELINE')
    parser.add_argument('--samples_dir_name', default='fid_samples_50k')

    # Number of samples and batch size
    parser.add_argument('--batch_size', type=int, default=1500)
    parser.add_argument('--n_samples', type=int, default=50000)
    args = parser.parse_args()

    # Get ckpt directory
    save_dir = os.path.join(args.output_dir, args.adversarial_loss_mode, args.setup_name)
    
    if os.path.exists( os.path.join(save_dir, 'checkpoints') ):
        ckpt_files = glob.glob( os.path.join(save_dir, 'checkpoints/*') )

        # Extract fid part from saved checkpoints
        fids = [ float(i.split('-')[1].split('=')[-1].split('.')[0]) for i in ckpt_files ]
        best_fid_index = np.argmin(fids)
        print("FID List ==> ", fids)
        print("Best FID index = {}, Best FID = {}".format(fids[best_fid_index], best_fid_index) )

        resume_from_checkpoint = ckpt_files[best_fid_index]
    else:
        raise Exception

    # Create GAN model
    model = ConvGenerator(args.z_dim, dim=args.dim,
                 n_upsamplings=args.n_upsamplings, setup_name=args.setup_name)
    
    # Load ckpt and get generator weights
    ckpt = torch.load(resume_from_checkpoint)['state_dict']
    G_state_dict = {k.replace('G.', ''): v for k, v in ckpt.items() if k.startswith('G')} # Map keys
    model.load_state_dict(G_state_dict)
    
    if not os.path.exists(os.path.join(save_dir, args.samples_dir_name)):
        # Generate samples
        generate_samples( model, args.z_dim, args.batch_size, args.n_samples, save_dir, args.samples_dir_name )


if __name__ == "__main__":
    main()