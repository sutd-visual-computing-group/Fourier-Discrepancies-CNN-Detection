# GAN Training 



## Hyper-parameters

For all reported CelebA experiments, 

- batch size = 64
- adam optimizer (β1=0.5, β2=0.999). 
- initial learning rate = 1e-4 for DCGAN and LSGAN, and initial learning rate = 2e-4 for WGAN-GP experiments.
- #epochs = 50 for DCGAN and LSGAN, and #epochs = 100 for WGAN-GP experiments.
- We use early stopping criterion using FID callback measured using 10k samples. 



For reported StarGAN experiments, we use default hyper-parameters used in the official repository



## Running the code

1. Create a new virtual environment and install all the dependencies

   ` pip install -r requirements.txt`

2. Create CelebA dataset (we use webdataset) 

   `python dataset_tool.py --img_dir <path_to_celeba_images> --save_dir <location_to_save_tar_files> --max_tar <maximum_number_of_tar_files> --dataset_name <celeba> --create_fid_10k_set <True>`

3. To train GAN models,

   1. For DCGAN, run `bash bash_scripts/dcgan_train_celeba.sh`
   2. For LSGAN, run `bash bash_scripts/lsgan_train_celeba.sh`
   3. For WGAN-GP, run `bash bash_scripts/wgan-gp_train_celeba.sh`

   

   

   

   