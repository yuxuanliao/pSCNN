# pSCNN
Nuclear magnetic resonance (NMR) spectroscopy provides us a powerful tool to analyze mixtures consisting of small molecules but is difficult to identify compounds in mixtures because of chemical shift variation between samples and peak overlapping among molecules. We presented a pseudo Siamese convolutional neural network method (pSCNN) to solve the problems of compound identification in NMR spectra of mixtures. This is the code repo for the paper *Highly accurate and large-scale collision cross section prediction with graph neural network for compound identification*.  

<div align="center">
<img src="https://raw.githubusercontent.com/yuxuanliao/pSCNN/main/Schematic_diagram_of_pSCNN.jpg" width=1063 height=912 />
</div>


# Installation

python and TensorFlow:

Python 3.7.10 and TensorFlow (version 2.1.0-GPU)

The main packages can be seen in [requirements.txt](https://github.com/yuxuanliao/pSCNN/blob/main/requirements.txt)

# Clone the repository and run it directly

[git clone at：https://github.com/xiaqiong/DeepCID.git](https://github.com/xiaqiong/DeepCID.git) 

# Download the model and run directly

Since the model exceeded the limit, we have uploaded all the models and the  information of mixtures to the Baidu SkyDrive and Google driver.

Download at: [Baidu SkyDrive](https://pan.baidu.com/s/1I0WMEvKvPNicy-i4Ru6uHQ) or [Google driver](https://drive.google.com/drive/folders/1DzMqiJRPDaLn2PcFW_myY_p0PO_VVEpS?usp=sharing)

**1.Training your model**

Run the file 'one-component-model.py'.The corresponding example data have been uploaded to the folder named 'augmented data'.

**2.Predict mixture spectra data**

Run the file 'DeepCID.py'.An example mixture data have been uploaded at Baidu SkyDrive (named  'mixture.npy', 'label.npy' and 'namedata.csv').Download the model and these example data，DeepCID can be reload and predict easily.

# Paper
[Paper](https://pubs.rsc.org/en/content/articlehtml/2019/an/c8an02212g)

# Contact

Zhi-Min Zhang: zmzhang@csu.edu.cn

