# pSCNN
Nuclear magnetic resonance (NMR) spectroscopy provides us a powerful tool to analyze mixtures consisting of small molecules but is difficult to identify compounds in mixtures because of chemical shift variation of the same compound in different mixtures and peak overlapping among molecules. We presented a pseudo Siamese convolutional neural network method (pSCNN) to solve the problems of compound identification in NMR spectra of mixtures. This is the code repo for the paper *Deep Learning-based Method for Compound Identification in NMR Spectra of Mixtures*.  

<div align="center">
<img src="https://github.com/yuxuanliao/pSCNN/blob/main/data augmentation and the pSCNN method.png" width=917 height=788 />
</div>


# Installation

Python and TensorFlow:

Python 3.8.13 and TensorFlow (version 2.5.0-GPU)

The main packages can be seen in [requirements.txt](https://github.com/yuxuanliao/pSCNN/blob/main/requirements/pip/requirements.txt)

- Install Anaconda
  https://www.anaconda.com/


- Install main packages in requirements.txt with following commands 

	```shell
	conda create --name pSCNN python=3.8.13
	conda activate pSCNN
	python -m pip install -r requirements.txt
	pip install tqdm
	```


# Clone the repo and run it directly

[git clone at：https://github.com/yuxuanliao/pSCNN.git](https://github.com/yuxuanliao/pSCNN.git)

# Download the model and run directly

Since the model exceeded the limit, we have uploaded the model and all the NMR data to Zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6504814.svg)](https://doi.org/10.5281/zenodo.6504814)


**Training your model and predict mixture spectra data**

Run the file 'pSCNN.py'. The model and these data have been uploaded at Zenodo. Download the model and these example data, pSCNN can be reload and predict easily.

# Contact

Yuxuan Liao: 232303012@csu.edu.cn
