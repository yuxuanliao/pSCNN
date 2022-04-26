# pSCNN
Nuclear magnetic resonance (NMR) spectroscopy provides us a powerful tool to analyze mixtures consisting of small molecules but is difficult to identify compounds in mixtures because of chemical shift variation between samples and peak overlapping among molecules. We presented a pseudo Siamese convolutional neural network method (pSCNN) to solve the problems of compound identification in NMR spectra of mixtures. This is the code repo for the paper *Highly accurate and large-scale collision cross section prediction with graph neural network for compound identification*.  

<div align="center">
<img src="https://github.com/yuxuanliao/pSCNN/blob/main/Schematic_diagram_of_pSCNN.png" width=917 height=788 />
</div>


# Installation

python and TensorFlow:

Python 3.7.10 and TensorFlow (version 2.1.0-GPU)

The main packages can be seen in [requirements.txt](https://github.com/yuxuanliao/pSCNN/blob/main/requirements.txt)

# Clone the repo and run it directly

[git clone at：https://github.com/yuxuanliao/pSCNN.git](https://github.com/yuxuanliao/pSCNN.git)

# Download the model and run directly

Since the model exceeded the limit, we have uploaded all the models and some NMR data to Zenodo.

[https://doi.org/10.5281/zenodo.5498987](https://doi.org/10.5281/zenodo.5498987)

**1.Training your model**

**1.Training your model**

Run the file 'dnn.py'.

**2.Predict mixture spectra data**

Run the file 'DeepCID.py'.An example mixture data have been uploaded at Baidu SkyDrive (named  'mixture.npy', 'label.npy' and 'namedata.csv').Download the model and these example data，DeepCID can be reload and predict easily.

# Contact

Yuxuan Liao: 212311021@csu.edu.cn


