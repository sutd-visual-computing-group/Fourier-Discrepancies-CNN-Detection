# General modules
import functools
from collections import OrderedDict
import math, os, PIL
import numpy as np
from tqdm import tqdm

# Pytorch modules
import torch
from torch import nn
import torchvision

# Lightning module
from pytorch_lightning import LightningModule

# FID module
from pytorch_fid.fid_score import calculate_fid_given_paths

# Import own modules here
import gan_loss
from gp import gradient_penalty
from settings import NUM_TRAINING_FILES
#from zero_insert_module import PadWithin



class Identity(LightningModule):
    """
    Identity mapping
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x



def _get_norm_layer_2d(norm):
    """
    Get 2D normalization layer
    """
    if norm == 'none':
        return Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError



class ConvGenerator(nn.Module):
    """
    Generic architecture for DCGAN, LSGAN and WGAN generators.
    """
    def __init__(   self,
                    input_dim: int = 128,
                    output_channels: int = 3,
                    dim: int = 32,
                    n_upsamplings: int = 5,
                    norm: str = 'batch_norm',
                    setup_name: str = 'BASELINE'):
        
        """
        Args:
            input_dim       : Latent space dimension
            output_channels : Number of output channels of GAN (Default 3 since RGB outputs)
            dim             : Useful to define number of channels at each layer
            n_upsamplings   : Number of upsampling (factor = 2) steps
            norm            : 2D normalization layer
            setup_name      : One of [ 'BASELINE', 'x.y.z' where x = {'N', 'Z', 'B'}, y = {'1', '3'}, z = {'3', '5', '7'} ]. Refer to paper for more details.
        """
        
        super().__init__()
        Norm = _get_norm_layer_2d(norm)

        def dconv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False or Norm == Identity),
                Norm(out_dim),
                nn.ReLU()
            )

        layers = []

        # 1: 1x1 -> 4x4
        d = min(dim * 2 ** (n_upsamplings - 1), dim * 16) # This is the major difference that needs to be fixed
        layers.append(dconv_norm_relu(input_dim, d, kernel_size=4, stride=1, padding=0))

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_upsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
            layers.append(dconv_norm_relu(d_last, d, kernel_size=4, stride=2, padding=1))


        # ------- Define last layer based on setup name ---------------
        # Kernel size = 5
        if setup_name == 'BASELINE':
            layers.append(nn.ConvTranspose2d(d, output_channels, kernel_size=4, stride=2, padding=1))
        
        elif setup_name == 'N.1.5':
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=5, stride=1, padding=2))
        
        elif setup_name == 'Z.1.5':
            layers.append(PadWithin(stride=2))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=5, stride=1, padding=2))
        
        elif setup_name == 'B.1.5':
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=5, stride=1, padding=2))
        
        
        # Kernel size = 3
        elif setup_name == 'N.1.3':
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=3, stride=1, padding=1))
        
        elif setup_name == 'Z.1.3':
            layers.append(PadWithin(stride=2))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=3, stride=1, padding=1))
        
        elif setup_name == 'B.1.3':
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=3, stride=1, padding=1))
        
        
        # Kernel size = 7
        elif setup_name == 'N.1.7':
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=7, stride=1, padding=3))
        
        elif setup_name == 'Z.1.7':
            layers.append(PadWithin(stride=2))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=7, stride=1, padding=3))
        
        elif setup_name == 'B.1.7':
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=7, stride=1, padding=3))

        
        # x.3.5 setups
        elif setup_name == 'N.3.5':
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=5, stride=1, padding=2))
        
        elif setup_name == 'Z.3.5':
            layers.append(PadWithin(stride=2))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=5, stride=1, padding=2))
        
        elif setup_name == 'B.3.5':
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, d, kernel_size=5, stride=1, padding=2))
            layers.append(nn.Conv2d(d, output_channels, kernel_size=5, stride=1, padding=2))

        else:
            raise Exception

        
        # Include Tanh as last layer for all setups
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)


    def forward(self, z):
        x = self.net(z)
        return x
        
    

class ConvDiscriminator(nn.Module):
    """
    Generic architecture for DCGAN, LSGAN and WGAN discriminators
    """
    def __init__(   self,
                    input_channels: int = 3,
                    dim: int = 32,
                    n_downsamplings: int = 5,
                    norm='batch_norm'):

        """
        Args:
            input_channels  : Number of input channels of GAN (Default 3 since RGB inputs)
            dim             : Useful to define number of channels at each layer
            n_downsamplings : Number of downsampling steps
            norm            : 2D normalization layer
        """

        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False or Norm == Identity),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )

        layers = []

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 16)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))

        # 2: logit
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y



class GAN(LightningModule):
    """
    Generic Pytorch Lightning GAN module for training.
    """
    
    def __init__(self,
                 latent_dim: int = 128,
                 dim: int = 32,
                 n_upsamplings=5,
                 lr: float = 2e-4,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64,
                 setup_name='BASELINE',
                 adversarial_mode: str = 'wgan',
                 gradient_penalty_mode: str = '0-gp',
                 gradient_penalty_sample_mode: str = 'line',
                 gradient_penalty_weight: float = 10.0,
                 gradient_penalty_d_norm: str = 'layer_norm',
                 n_d: int = 5,
                 sample_every: int=100,
                 expected_epochs: int=-1,
                 expected_dataset: str = 'celeba',
                 fid_measure_samples: int=10000,
                 fid_ref_data_dir: str = 'datasets/celeba/fid_eval_10k/',
                 inference_batch_size: int = 2000,
                 **kwargs):
        
        """
        Args:
            lr                          : initial learning rate
            b1                          : beta1 for Adam optimizer
            b2                          : beta2 for Adam optimizer
            batch_size                  : Batch size
            adversarial_mode            : Loss mode
            gradient_penalty_mode       : gradient penalty
            gradient_penalty_sample_mode: gp_sample_mode
            gradient_penalty_weight     : gp_weight
            gradient_penalty_d_norm     : discriminator normalization
            n_d                         : number of discriminator updates per generator update
            sample_every                : frequency to sample images during training
            expected_epochs             : number of max epochs to train
            expected_dataset            : dataset
            fid_measure_samples         : number of samples to measure fid after every epoch (We use 10k images)
            fid_ref_data_dir            : directory with dataset images to measure FID (We use 10k images)
            inference_batch_size        : batch size to generate samples during training
        """

        super().__init__()
        self.save_hyperparameters()

        # ---- Attributes ----
        # > GAN attributes
        self.latent_dim = latent_dim
        self.adversarial_loss_mode = adversarial_mode
        self.d_loss_fn, self.g_loss_fn = self.get_loss_fns()
        
        self.gradient_penalty_mode = gradient_penalty_mode
        self.gradient_penalty_sample_mode = gradient_penalty_sample_mode
        self.gradient_penalty_weight = gradient_penalty_weight
        self.gradient_penalty_d_norm = gradient_penalty_d_norm
        self.d_norm = self.get_d_norm()
        

        # > Optimizer attributes
        self.b1 = b1
        self.b2 = b2
        self.lr = lr

        # > Training attributes
        self.batch_size = batch_size
        self.n_d = n_d
        self.sample_every = sample_every
        
        # Create Generator and Discriminator
        self.G = ConvGenerator(latent_dim, dim=dim,
                 n_upsamplings=n_upsamplings, setup_name=setup_name)
        print(self.G) # print for confirmation of architecture
        self.D = ConvDiscriminator(dim=dim, n_downsamplings=n_upsamplings, norm=self.d_norm)

        # Constant noise for evaluation duing training
        self.noise = torch.randn(100, self.latent_dim, 1, 1)

        # Additional arguments to fix the learning rate schedulers
        self.expected_epochs = expected_epochs
        self.expected_dataset_name = expected_dataset

        # Metrics (Max is 50000)
        self.fid_measure_samples = min(50000, fid_measure_samples)
        self.fid_ref_data_dir = fid_ref_data_dir
        self.inference_batch_size = inference_batch_size # This is used for generating data for FID and half of this batch size to measure FID


    def configure_optimizers(self):
        opt_g = torch.optim.Adam( self.G.parameters(), betas=(self.b1, self.b2), lr=self.lr )
        opt_d = torch.optim.Adam( self.D.parameters(), betas=(self.b1, self.b2), lr=self.lr )

        # Setup schedulers
        gamma = -np.log(1e-6/self.lr)/ self.expected_epochs
        factor = np.exp(-gamma)

        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_g, mode='min', factor=factor, patience=1, \
                        threshold=1.00, threshold_mode='abs', cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, mode='min', factor=factor, patience=1, \
                        threshold=1.00, threshold_mode='abs', cooldown=0, min_lr=1e-6, eps=1e-08, verbose=True)
        
        return [opt_g, opt_d],\
                [{'scheduler': g_scheduler, 'monitor': 'fid10k', 'interval': 'epoch', 'name':'G_lr', 'reduce_on_plateau': True},
                {'scheduler': d_scheduler, 'monitor': 'fid10k', 'interval': 'epoch', 'name':'D_lr', 'reduce_on_plateau': True}]


    def get_loss_fns(self):
        d_loss_fn, g_loss_fn = gan_loss.get_adversarial_losses_fn(self.adversarial_loss_mode)
        return d_loss_fn, g_loss_fn


    def get_d_norm(self):
        # Setup discriminator norm (Cannot use batch normalization with gradient penalty)
        if self.gradient_penalty_mode == 'none':
            d_norm = 'batch_norm'
        else:
            d_norm = self.gradient_penalty_d_norm
        return d_norm


    def forward(self, z):
        return self.G(z)

    
    def training_step(self, batch, batch_idx, optimizer_idx):
        x_real = batch[0]

        # sample noise
        z = torch.randn(x_real.shape[0], self.latent_dim, 1, 1)
        z = z.type_as(x_real)

        # train generator
        if optimizer_idx == 0 and batch_idx%self.n_d == 0:
            
            # Set to train mode
            self.G.train()
            self.D.train()

            # Generate fake images
            x_fake = self(z)

            # Calculate discriminator values
            x_fake_d_logit = self.D(x_fake)

            # Calculate Generator loss
            g_loss = self.g_loss_fn(x_fake_d_logit)

            if batch_idx % self.sample_every == 0:
                self._sample_during_training(batch_idx)

            self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return g_loss


        # train discriminator
        if optimizer_idx == 1:
            # Set to train mode
            self.G.train()
            self.D.train()
            
            # Generate fake images
            x_fake = self.G(z).detach()

            # Calculate discriminator values
            x_real_d_logit = self.D(x_real)
            x_fake_d_logit = self.D(x_fake)

            # Calculate discriminator loss
            x_real_d_loss, x_fake_d_loss = self.d_loss_fn(x_real_d_logit, x_fake_d_logit)
            gp = gradient_penalty(functools.partial(self.D), x_real, x_fake, gp_mode=self.gradient_penalty_mode, \
                        sample_mode=self.gradient_penalty_sample_mode)

            # Calculate Discriminator loss
            d_loss = (x_real_d_loss + x_fake_d_loss) + gp * self.gradient_penalty_weight

            self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log('gp', gp, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            return d_loss

        return


    def training_epoch_end(self, *args, **kwargs):
        gen_samples_dir = os.path.join(self.trainer.log_dir, 'fid_samples_10k')
        os.makedirs(gen_samples_dir, exist_ok=True)
        
        self.generate_samples_for_fid(self.fid_measure_samples, gen_samples_dir)
        torch.cuda.empty_cache()

        # Calculate FID
        paths = [ self.fid_ref_data_dir, gen_samples_dir ]
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        fid = calculate_fid_given_paths(paths,
                                self.inference_batch_size//4,
                                device,
                                2048)
        self.log('fid10k', fid, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def generate_samples_for_fid(self, n_samples, gen_samples_dir):
        self.G.eval()

        for i in range( int(math.ceil(n_samples/self.inference_batch_size)) ):
            # Generate iteratively
            z = torch.randn( min( self.inference_batch_size, int(n_samples-self.inference_batch_size*i) ), self.latent_dim, 1, 1)
            z = z.type_as(self.G.net[0][0].weight)
            samples = self.G(z).detach()

            self.save_samples_as_png(samples, gen_samples_dir, i*self.inference_batch_size)
            del samples, z
            torch.cuda.empty_cache()


    def save_samples_as_png(self, samples_tensor, save_loc, start_index):
        #samples_tensor = samples_tensor.cpu().numpy()
        std = [0.5, 0.5, 0.5]
        mean = [0.5, 0.5, 0.5]

        for i in range(samples_tensor.shape[0]):
            index = i + start_index
            img = (samples_tensor[i, :, :, :].permute(1, 2, 0).cpu().numpy()*std) + mean
            img_pil = PIL.Image.fromarray(np.uint8(img*255.0))
            img_pil.save(os.path.join(save_loc, "{}.png".format(index)), quality=95, subsampling=-1)

        return


    def test_step(self):
        return self.sample()


    def _sample_during_training(self, batch_idx):
        self.G.eval()
        z = self.noise.type_as(self.G.net[0][0].weight)

        # Directory for saving
        self.samples_dir = os.path.join(self.trainer.log_dir, 'training_samples')
        os.makedirs(self.samples_dir, exist_ok = True)

        # log sampled images
        sample_imgs = (self(z)*0.5) + 0.5
        grid = torchvision.utils.make_grid(sample_imgs, nrow=10, normalize=False)
        torchvision.utils.save_image(sample_imgs, nrow=10, fp='{}/sample_ep={}_it={}.jpg'.format(self.samples_dir,\
            self.current_epoch, batch_idx))
        torch.cuda.empty_cache()


    def get_progress_bar_dict(self):
        r"""
        Disable loss logging since G and D losses are different
        """
        # call .item() only once but store elements without graphs
        tqdm_dict = {}

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            version = self.trainer.logger.version
            version = version if isinstance(version, str) else version
            tqdm_dict['gan_setup'] = version

        return tqdm_dict



if __name__ == "__main__":
    #pass
    g = ConvGenerator()
    print(g.net[0][0].weight)

    d = ConvDiscriminator()
    print(d)

    dcgan = GAN()
    print(dcgan)
