# Fourier Synthetic Image Classifiers

## About the Code

The section uses high frequency features extracted using the official matlab implementation by Dzanic et al. [[1]](#1). The matlab code can be found [[2]](#2)



This codebase leverages on Pytorch Lightning [[1]](#1) module to efficiently scale our GAN training to multi-GPU infrastructure. We also use Webdataset [[2]](#2) (Pytorch Iterable Dataset implementation) that allows efficient access to datasets stored in POSIX tar archives using only sequential disk access. We also provide the Docker file to run our code. The codebase is clearly documented with bash file execution points exposing all required arguments and hyper-parameters.

- [x] Pytorch
- [x] Pytorch Lightning [[1]](#1)
- [x] WebDataset [[2]](#2)
- [x] Multi-GPU training
- [x] DockerFile



## Hyper-parameters

For all reported CelebA/ LSUN/ StarGAN experiments, 

- For feature extraction, we use the official script. We tried with different hyper-parameters, but in all instances there was no substantial variation in classifier bypassing values.
- All results reported are averaged across 10 independent runs
- For KNN classifier, we use k=5 similar to official implementation
- For SVM, we use RBF kernel
- For MLP, we use a 2-layer architecture with sigmoid activation functions (Binary classification learning problem).



## Running the Code

1. Create a new virtual environment and install all the dependencies

   ` pip install -r requirements.txt`



2. Use the `generateddata.m` script from the official repository [[2]](#2) to extract the high frequency features. (3 dimensional features)



3. To run the classifiers, first organize the directory as follows using the fits obtained.

   fits/

      \-  celeba/

   ​     \- real/

   ​     \- gan/

   ​       \- BASELINE/

   ​       \- N.1.5/

   ​       \- Z.1.5/

   ​       \- ..../

   ​     \- lsgan/

   ​       \- BASELINE/

   ​       \- N.1.5/

   ​       \- Z.1.5/

   ​       \- ..../

   ​     \- wgan/

   ​       \- BASELINE/

   ​       \- N.1.5/

   ​       \- Z.1.5/

   ​       \- ..../

   ```markdown
   var routes = (
     <Route name="fits">
       <Route name="real">
       </Route>
       <Route name="gan">
         <Route name="BASELINE"/>
       </Route>
     </Route>
   );
   ```



1. To train GAN models,
   - For DCGAN, run `bash bash_scripts/dcgan_train_celeba.sh`
   - For LSGAN, run `bash bash_scripts/lsgan_train_celeba.sh`
   - For WGAN-GP, run `bash bash_scripts/wgan-gp_train_celeba.sh`



## CelebA FID scores

|          | DCGAN | LSGAN | WGAN-GP |
| -------- | ----- | ----- | ------- |
| Baseline | 88.6  | 73.26 | 60.6    |
| N.1.5    | 87.52 | 70.69 | 48.69   |
| Z.1.5    | 69.14 | 60.29 | 47.73   |
| B.1.5    | 84.65 | 78.66 | 52.18   |
| N.1.7    | 90.8  | 73.09 | 60.11   |
| Z.1.7    | 71.45 | 59.55 | 43.1    |
| B.1.7    | 79.92 | 76.33 | 55.28   |
| N.1.3    | 93.54 | 74.06 | 58.35   |
| Z.1.3    | 65.46 | 61.45 | 56.91   |
| B.1.3    | 76.04 | 81.97 | 58.55   |
| N.3.5    | 73.63 | 78.31 | 55.47   |
| Z.3.5    | 68.41 | 66.27 | 57.59   |
| B.3.5    | 80.89 | 72.29 | 54.84   |



## StarGAN Experiments

For reported StarGAN experiments, we use the official repository [[3]](#3) with default hyper-parameters. We include a script illustrating on how to change last feature map scaling for StarGAN Generator architecture.  Refer to `stargan_v1_model.py`



## Spectral Regularization Experiments

For spectral regularization, we use the official repository [[4]](#4) with default hyper-parameters.



## References

<a id="1">[1]</a> Fourier Spectrum Discrepancies in Deep Network Generated Images, Tarik Dzanic, Karan Shah, and Freddie D. Witherden, In NeurIPS, 2020

<a id="2">[2]</a> https://github.com/tarikdzanic/FourierSpectrumDiscrepancies)

<a id="3">[3]</a> https://github.com/yunjey/stargan

<a id="4">[4]</a> https://github.com/cc-hpc-itwm/UpConv




