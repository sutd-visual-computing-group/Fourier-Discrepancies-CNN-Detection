<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                A Closer Look at Fourier Spectrum Discrepancies for</br>CNN-generated Images Detection</h1>
<p align='center' style="text-align:center;font-size:1.25em;">
    Keshigeyan&nbsp;Chandrasegaran&nbsp;/&nbsp;
    Ngoc&#8209;Trung&nbsp;Tran&nbsp;/&nbsp;
    Ngai&#8209;Man&nbsp;Cheung</br>
Singapore University of Technology and Design (SUTD)<br/>
<em>To appear in CVPR&nbsp;2021&nbsp;(Oral)</br></em>
<a href="https://keshik6.github.io/Fourier-Discrepancies-CNN-Detection/" title="Project" target="_blank" rel="nofollow">Project</a> |
<a href="https://arxiv.org/abs/2103.17195" title="CVPR Paper" target="_blank" rel="nofollow">CVPR Paper</a> |
<a href="https://drive.google.com/drive/folders/123RpZhytXBrJQyHg_0f46W-Qv3t5Hwsl?usp=sharing" title="GAN Models" target="_blank" rel="nofollow">GAN Models</a>
</p>



## Abstract

CNN-based generative modelling has evolved to produce synthetic images indistinguishable from real images in the RGB pixel space. Recent works have observed that CNN-generated images share a systematic shortcoming in replicating high frequency Fourier spectrum decay attributes. Furthermore, these works have successfully exploited this systematic shortcoming to detect CNN-generated images reporting up to 99% accuracy across multiple state-of-the-art GAN models.

</br>

In this work, we investigate the validity of assertions claiming that CNN-generated images are unable to achieve high frequency spectral decay consistency. We meticulously construct a counterexample space of high frequency spectral decay consistent CNN-generated images emerging from our handcrafted experiments using DCGAN, LSGAN, WGAN-GP and StarGAN, where we empirically show that this frequency discrepancy can be avoided by a minor architecture change in the last upsampling operation. We subsequently use images from this counterexample space to successfully bypass the recently proposed forensics detector which leverages on high frequency Fourier spectrum decay attributes for CNN-generated image detection.

</br>

Through this study, we show that high frequency Fourier spectrum decay discrepancies are not inherent characteristics for existing CNN-based generative models--contrary to the belief of some existing work--, and such features are not robust to perform synthetic image detection. Our results prompt re-thinking of using high frequency Fourier spectrum decay attributes for CNN-generated image detection.



<img src="/assets/web.jpg" />



## About the code

**GAN  :** This is written in Pytorch. This codebase leverages on Pytorch Lightning [[1]](#1) module to efficiently scale our GAN training to multi-GPU infrastructure. We also use Webdataset [[2]](#2) (Pytorch Iterable Dataset implementation) that allows efficient access to datasets stored in POSIX tar archives using only sequential disk access. We also provide the Docker file to run our code. The codebase is clearly documented with bash file execution points exposing all required arguments and hyper-parameters.



**Fourier Synthetic Image classifier :** The code uses high frequency features extracted using the official matlab implementation by Dzanic et al. [[3]](#3). The matlab code can be found [[4]](#4). The code is clearly documented.



## Running the code

**GAN :** Clear steps on how to run and reproduce our results for DCGAN, LSGAN, WGAN-GP and StarGAN experiments can be found at [src/gans/README.md](src/gans/README.md)



**Fourier Synthetic Image classifier :** Clear steps on how to run and reproduce our results for KNN, SVM and MLP classfier experiments using high frequency features extracted from Dzanic et al work can be found at [src/classifiers/README.md](src/classifiers/README.md)



## GAN Samples

|   GAN   |                Baseline                |                N.1.5                |                B.1.5                |                Z.1.5                |
| :-----: | :------------------------------------: | :---------------------------------: | :---------------------------------: | :---------------------------------: |
|  DCGAN  |  <img src="/assets/gan_Baseline.png">  |  <img src="/assets/gan_N.1.5.png">  |  <img src="/assets/gan_B.1.5.png">  |  <img src="/assets/gan_Z.1.5.png">  |
|  LSGAN  | <img src="/assets/lsgan_Baseline.png"> | <img src="/assets/lsgan_N.1.5.png"> | <img src="/assets/lsgan_B.1.5.png"> | <img src="/assets/lsgan_Z.1.5.png"> |
| WGAN-GP | <img src="/assets/wgan_Baseline.png">  | <img src="/assets/wgan_N.1.5.png">  | <img src="/assets/wgan_B.1.5.png">  | <img src="/assets/wgan_Z.1.5.png">  |



## Citation

```markdown
@InProceedings{Chandrasegaran2021,
    author        = {Keshigeyan Chandrasegaran and Ngoc-Trung Tran and Ngai-Man Cheung},
    booktitle     = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    title         = {A Closer Look at Fourier Spectrum Discrepancies for CNN-generated Images Detection},
    year          = {2021}
}
```



## Acknowledgements

We gratefully acknowledge the following works:

- https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch
- https://github.com/mseitzer/pytorch-fid



## References

<a id="1">[1]</a> https://www.pytorchlightning.ai/

<a id="2">[2]</a> https://github.com/tmbdev/webdataset

<a id="3">[3]</a> Fourier Spectrum Discrepancies in Deep Network Generated Images, Tarik Dzanic, Karan Shah, and Freddie D. Witherden, In NeurIPS, 2020

<a id="4">[4]</a> https://github.com/tarikdzanic/FourierSpectrumDiscrepancies)

