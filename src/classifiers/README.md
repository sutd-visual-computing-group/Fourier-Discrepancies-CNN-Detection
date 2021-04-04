# Fourier Synthetic Image Classifiers

## About the Code

The section uses high frequency features extracted using the official matlab implementation by Dzanic et al. [[1]](#1). The matlab code can be found [[2]](#2). The code is clearly documented.



## Hyper-parameters

For all reported CelebA/ LSUN/ StarGAN experiments, 

- For feature extraction, we use the official script. We tried with different hyper-parameters, but in all instances there was no substantial variation in classifier bypassing values.
- All results reported are averaged across 10 independent runs
- For KNN classifier, we use k=5 as official implementation
- For SVM, we use RBF kernel
- For MLP, we use a 2-layer architecture with sigmoid activation functions (Binary classification learning problem).



## Running the Code

1. Create a new virtual environment and install all the dependencies

   ` pip install -r requirements.txt`



2. Use the `generateddata.m` script from the official repository [[2]](#2) to extract the high frequency features. (3 dimensional features)



3. To run the classifiers, first organize the directory as follows and place the `.mat` files inside

~~~bash
```
├── fits
│   ├── celeba
│   │   ├── real
│   │   ├── gan
│   │   ├──   ├── BASELINE
│   │   ├──   ├── B.1.5
│   │   ├──   ├── Z.1.5
│   │   ├──   ├── ....

│   │   ├── lsgan
│   │   ├──   ├── BASELINE
│   │   ├──   ├── B.1.5
│   │   ├──   ├── Z.1.5
│   │   ├──   ├── ....

│   │   ├── wgan
│   │   ├──   ├── BASELINE
│   │   ├──   ├── B.1.5
│   │   ├──   ├── Z.1.5
│   │   ├──   ├── ....
```
~~~

4. To train and test with classifiers,

- use `classifiers/knn_classifier.py` for KNN classification
- use `classifiers/svm_classifier.py` for SVM classification
- use `classifiers/mlp_classifier.py` for MLP classification



## References

<a id="1">[1]</a> Fourier Spectrum Discrepancies in Deep Network Generated Images, Tarik Dzanic, Karan Shah, and Freddie D. Witherden, In NeurIPS, 2020

<a id="2">[2]</a> https://github.com/tarikdzanic/FourierSpectrumDiscrepancies)




