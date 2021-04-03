# GAN Training



## Hyper-parameters

For all reported CelebA experiments, 

- batch size = 64
- adam optimizer (β1=0.5, β2=0.999). 
- initial learning rate = 1e-4 for DCGAN and LSGAN, and initial learning rate = 2e-4 for WGAN-GP experiments.
- #epochs = 50 for DCGAN and LSGAN, and #epochs = 100 for WGAN-GP experiments.
- We use early stopping criterion using FID callback measured using 10k samples. 



For reported StarGAN experiments, we use default hyper-parameters used in the official repository



