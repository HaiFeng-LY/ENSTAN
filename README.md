# Spatiotemporal Alignment for EEG-fNIRS Emotion Recognition
![image](https://github.com/HaiFeng-LY/ENSTAN/blob/main/DyGAT.png)
# Abstract  
Due to neurovascular coupling, an inherent emotional complementarity is observed between electroencephalography (EEG) and functional near-infrared spectroscopy (fNIRS); however, these modalities exhibit significant cross-modal spatiotemporal and individual differences in emotional expression. In addition, fNIRS reveals cross-categorical emotional variations resulting from the relative responses of oxygenated hemoglobin and deoxyhemoglobin. To address these challenges, this paper proposes the EEG-fNIRS spatiotemporal alignment network (ENSTAN), which comprises a dynamic multigraph attention mechanism, temporally aligned cross-attention, and a cross-category source-target emotion alignment module. Given the limited availability of publicly accessible EEG-fNIRS emotion datasets, we validated the proposed ENSTAN on the self-constructed ENTER dataset and compared it with various baseline methods. The results demonstrate that ENSTAN achieved average accuracies of 91.97\% in random cross-validation and 88.96\% in trial-unit cross-validation for subject-dependent experiments, with a 73.24\% average accuracy in subject-independent experiments, confirming its effectiveness.In addition, the proposed ENSTAN elucidates the salient interaction channels and response latencies between EEG and fNIRS.

# EEG-NIRS dataset TYUT emotion recognition (ENTER)
https://gitee.com/tycgj/enter
![image](https://github.com/HaiFeng-LY/ENSTAN/blob/main/EEG-fNIRS.png)
# Environment Setup
Create the environment and install dependencies:
*torch>=2.5.1
*torchvision>=0.20.1
*torchaudio>=2.5.1
*numpy>=1.26.4
*scipy>=1.14.0
*scikit-learn>=1.6.1
*matplotlib>=3.9.0
*seaborn>=0.13.2
*pandas>=2.2.2
*tqdm>=4.67.1
*mne>=1.2.0
*torch-geometric>=2.1.0
*torch-scatter>=2.1.2
*torch-sparse>=0.6.18
*torch-cluster>=1.6.3
*torch-spline-conv>=1.2.2
*opencv-python>=4.10.0
*Pillow>=11.1.0
*statsmodels>=0.14.5
*xlrd>=2.0.1
*openpyxl>=3.1.5
*scikit-image>=0.25.0


# Running Experiments


*3412

`python







