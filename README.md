# Spatiotemporal Alignment for EEG-fNIRS Emotion Recognition
This is the official implementation of NESTAN, the model proposed in our paper “Spatiotemporal Alignment for EEG-fNIRS Emotion Recognition”. NESTAN is a novel spatiotemporal alignment framework designed for robust cross-modal fusion and representation learning from EEG and fNIRS signals.
# Abstract  
Due to neurovascular coupling, an inherent emotional complementarity is observed between electroencephalography (EEG) and functional near-infrared spectroscopy (fNIRS); however, these modalities exhibit significant cross-modal spatiotemporal and individual differences in emotional expression. Furthermore, there is an inherent negative coupling between oxygenated hemoglobin and deoxygenated hemoglobin in fNIRS, which can lead to inconsistent results when models perform emotion prediction based on both.
To address these challenges, this paper proposes the EEG-fNIRS spatiotemporal alignment network (ENSTAN), which comprises a dynamic multigraph attention mechanism, temporally aligned cross-attention, and a subspace source-target emotion alignment module. Given the limited availability of publicly accessible EEG-fNIRS emotion datasets, we validated the proposed ENSTAN on the self-constructed ENTER dataset and compared it with various baseline methods. The results demonstrate that ENSTAN achieved average accuracies of 91.97% in random cross-validation and 88.96% in trial-unit cross-validation for subject-dependent experiments, with a 73.24% average accuracy in subject-independent experiments, confirming its effectiveness. In addition, the proposed ENSTAN elucidates the salient interaction channels and response latencies between EEG and fNIRS.

# EEG-fNIRS TYUT Emotion Recognition  dataset (ENTER)
https://gitee.com/tycgj/enter

## Experimental Paradigm of the ENTER Dataset
The ENTER dataset includes 60 emotional video clips covering four discrete emotions: sadness, happiness, neutrality, and fear, with 15 clips for each category. The duration of the videos ranges from 1 to 2 minutes. All videos were pre-evaluated and screened by 20 non-participant individuals using the Self-Assessment Manikin (SAM) scale on arousal and valence dimensions to ensure effective emotion elicitation. Each trial followed a fixed sequence consisting of a 5-second preparation phase, a 5-second resting baseline, 1–2 minutes of emotional video presentation, a 30-second self-report assessment, and a 5-second rest period. The experiment comprised a total of 60 trials, and the video clips were presented in a randomized order.

## Detailed Data Information
| Feature                | Detail                                                                                                  |
|------------------------|----------------------------------------------------------------------------------------------------------|
| Number of Participants | 50 (gender-balanced, 1:1 ratio)                                                                          |
| Health Status          | Right-handed; normal hearing and vision; no neurological disorders                                       |
| EEG Configuration      | 62 electrodes, 1000 Hz sampling rate                                                                     |
| fNIRS Configuration    | 18 optodes (sources + detectors), 11 Hz sampling rate                                                    |
| Trial Task             | Participants watched an emotional video clip (~80 seconds)                                               |
| Self-report            | 30-second SAM rating after each video (valence and arousal)                                              |
| Emotions               | Sadness, happiness, neutrality, fear (15 videos per category)                                            |
| Total Trials per Subject | 60 trials                                                                                              |


## Topomap
<img src="https://github.com/HaiFeng-LY/ENSTAN/blob/main/EEG-fNIRS.png" width="500">

## Screened Emotion-Eliciting Videos


| Clip Number | Source Film                                   | Targeted Emotion | Duration (s) | Language |
|-------------|-----------------------------------------------|------------------|--------------|----------|
| 1           | Aftershock Tangshan Earthquake                | Sadness          | 87           | Chinese  |
| 2           | Aftershock Tangshan Earthquake                | Sadness          | 95           | Chinese  |
| 3           | The Bravest                                   | Sadness          | 73           | Chinese  |
| 4           | The Bravest                                   | Sadness          | 59           | Chinese  |
| 5           | Dying to Survive                              | Sadness          | 100          | Chinese  |
| 6           | Dying to Survive                              | Sadness          | 65           | Chinese  |
| 7           | A Little Red Flower                           | Sadness          | 47           | Chinese  |
| 8           | The Allure of Tears                           | Sadness          | 90           | Chinese  |
| 9           | Run For Young                                 | Sadness          | 81           | Chinese  |
| 10          | Heaven Calls                                   | Sadness          | 83           | Chinese  |
| 11          | A Little Reunion                              | Sadness          | 82           | Chinese  |
| 12          | A Little Reunion                              | Sadness          | 87           | Chinese  |
| 13          | With You                                      | Sadness          | 43           | Chinese  |
| 14          | With You                                      | Sadness          | 116          | Chinese  |
| 15          | With You                                      | Sadness          | 78           | Chinese  |
| 16          | Trump Card (Season V)                         | Happiness        | 113          | Chinese  |
| 17          | Trump Card (Season V)                         | Happiness        | 116          | Chinese  |
| 18          | Trump Card (Season V)                         | Happiness        | 82           | Chinese  |
| 19          | Trump Card (Season V)                         | Happiness        | 118          | Chinese  |
| 20          | Trump Card (Season V)                         | Happiness        | 57           | Chinese  |
| 21          | Trump Card (Season V)                         | Happiness        | 63           | Chinese  |
| 22          | Trump Card (Season V)                         | Happiness        | 117          | Chinese  |
| 23          | Trump Card (Season V)                         | Happiness        | 81           | Chinese  |
| 24          | Trump Card (Season V)                         | Happiness        | 54           | Chinese  |
| 25          | Trump Card (Season V)                         | Happiness        | 42           | Chinese  |
| 26          | Trump Card (Season V)                         | Happiness        | 83           | Chinese  |
| 27          | Trump Card (Season V)                         | Happiness        | 42           | Chinese  |
| 28          | Trump Card (Season V)                         | Happiness        | 56           | Chinese  |
| 29          | Trump Card (Season V)                         | Happiness        | 102          | Chinese  |
| 30          | Trump Card (Season V)                         | Happiness        | 60           | Chinese  |
| 31          | The Tale of Chinese Medicine (Season I)       | Calmness         | 58           | Chinese  |
| 32          | The Tale of Chinese Medicine (Season I)       | Calmness         | 90           | Chinese  |
| 33          | The Tale of Chinese Medicine (Season I)       | Calmness         | 84           | Chinese  |
| 34          | The Tale of Chinese Medicine (Season I)       | Calmness         | 92           | Chinese  |
| 35          | The Tale of Chinese Medicine (Season I)       | Calmness         | 100          | Chinese  |
| 36          | The Tale of Chinese Medicine (Season I)       | Calmness         | 60           | Chinese  |
| 37          | The Tale of Chinese Medicine (Season I)       | Calmness         | 63           | Chinese  |
| 38          | The Tale of Chinese Medicine (Season I)       | Calmness         | 95           | Chinese  |
| 39          | The Tale of Chinese Medicine (Season I)       | Calmness         | 68           | Chinese  |
| 40          | The Tale of Chinese Medicine (Season I)       | Calmness         | 70           | Chinese  |
| 41          | The Tale of Chinese Medicine (Season I)       | Calmness         | 71           | Chinese  |
| 42          | The Tale of Chinese Medicine (Season I)       | Calmness         | 90           | Chinese  |
| 43          | The Tale of Chinese Medicine (Season I)       | Calmness         | 87           | Chinese  |
| 44          | The Tale of Chinese Medicine (Season I)       | Calmness         | 86           | Chinese  |
| 45          | The Tale of Chinese Medicine (Season I)       | Calmness         | 77           | Chinese  |
| 46          | Bedfellows                                    | Fear             | 83           | English  |
| 47          | Annabelle                                     | Fear             | 56           | English  |
| 48          | Annabelle                                     | Fear             | 67           | English  |
| 49          | Lights Out                                    | Fear             | 85           | English  |
| 50          | A Quiet Place                                 | Fear             | 62           | English  |
| 51          | It                                            | Fear             | 112          | English  |
| 52          | Pictured                                       | Fear             | 61           | English  |
| 53          | Under the Bed                                 | Fear             | 51           | English  |
| 54          | The Shining                                   | Fear             | 75           | English  |
| 55          | Closet Space                                  | Fear             | 76           | English  |
| 56          | Unfriend                                       | Fear             | 73           | English  |
| 57          | Creeping                                       | Fear             | 76           | English  |
| 58          | Dead Air                                       | Fear             | 70           | English  |
| 59          | The Mime                                      | Fear             | 74           | English  |
| 60          | The Mime                                      | Fear             | 81           | English  |


# Environment Setup

Create the environment and install dependencies:

- torch >= 2.5.1
- torchvision >= 0.20.1
- torchaudio >= 2.5.1
- numpy >= 1.26.4
- scipy >= 1.14.0
- scikit-learn >= 1.6.1
- matplotlib >= 3.9.0
- seaborn >= 0.13.2
- pandas >= 2.2.2
- tqdm >= 4.67.1
- mne >= 1.2.0
- torch-geometric >= 2.1.0
- torch-scatter >= 2.1.2
- torch-sparse >= 0.6.18
- torch-cluster >= 1.6.3
- torch-spline-conv >= 1.2.2
- opencv-python >= 4.10.0
- Pillow >= 11.1.0
- statsmodels >= 0.14.5
- xlrd >= 2.0.1
- openpyxl >= 3.1.5
- scikit-image >= 0.25.0
























