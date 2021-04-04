# GAN Training 





## Hyper-parameters

For all reported CelebA experiments, 

- batch size = 64
- adam optimizer ($β_{1}$=0.5, $β_{2}$=0.999). 
- initial learning rate = 1e-4 for DCGAN and LSGAN, and initial learning rate = 2e-4 for WGAN-GP experiments.
- epochs = 50 for DCGAN and LSGAN, and #epochs = 100 for WGAN-GP experiments.
- We use early stopping criterion using FID callback measured using 10k samples. 



For reported StarGAN experiments, we use default hyper-parameters used in the official repository



## About the Code

The code is written in Pytorch. This codebase leverages on Pytorch Lightning[^1] module to efficiently scale our GAN training to multi-GPU infrastructure. We also use Webdataset (Pytorch Iterable Dataset implementation) that allows efficient access to datasets stored in POSIX tar archives using only sequential disk access. We also provide the DockerFile to run the GAN training in HPC systems. The codebase is clearly documented with bash file execution points exposing all required arguments and hyper-parameters.

- [x] Pytorch
- [x] Pytorch Lightning [^1]
- [x] WebDataset [^2]
- [x] Multi-GPU training
- [x] DockerFile



## Running the Code

Create a new virtual environment and install all the dependencies

` pip install -r requirements.txt`

Create CelebA tar archives dataset. We also require a reference set of 10k CelebA images to measure FID during training 

`python dataset_tool.py --img_dir <path_to_celeba_images> --save_dir <location_to_save_tar_files> --max_tar <maximum_number_of_tar_files> --dataset_name <celeba> --create_fid_10k_set <True>`

To train GAN models,

For DCGAN, run `bash bash_scripts/dcgan_train_celeba.sh`

For LSGAN, run `bash bash_scripts/lsgan_train_celeba.sh`

For WGAN-GP, run `bash bash_scripts/wgan-gp_train_celeba.sh`







## References

[^1]: https://www.pytorchlightning.ai/ 
[^2]: https://github.com/tmbdev/webdataset







