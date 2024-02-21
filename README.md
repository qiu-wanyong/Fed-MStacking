# Fed-MStacking
The Code of Fed-MStacking: Heterogeneous Federated Learning with Stacking Misaligned Labels for Abnormal Heart Sound  Detection

## Abstract
Cardiovascular diseases (CVDs) remain the leading cause of disease burden in the world. In high-income countries, cardiac imaging techniques and electrocardiogram are widely used to assist in the diagnosis of CVDs. However, on low- and middle-income countries with limited resources, for the above approaches it is often difficult to achieve widespread use. Heart sound auscultation is a cost-effective and non-invasive diagnostic method, which is more suitable for promotion on a wider area. This method can also reveal many pathological cardiac conditions, such as valvular disease (VD), coronary artery disease (CAD), and arrhythmias. Moreover, artificial intelligence (AI) technology can further reduce the impact of objective factors on disease detection in developing regions (e.\,g., low physician proficiency and insufficient number of physicians), and has great potential for early detection and remote monitoring of CVDs based on the Internet of Health Things (IoHT). Thus, it is particularly important to develop AI auscultation methods that are more efficient and suitable for real medical scenarios.

#### Index Terms— Ensemble learning, Federated learning, Heart sound, Misaligned labels, Privacy protection

## Data
Preparation and Analysis of the Signals for [PhysioNet/CinC Challenge Dataset:](https://physionet.org/content/challenge-2016/1.0.0/)

Since the study involves heterogeneous local models, as shown in Fig 1, we processed raw heart sound recordings based on two modalities. 

<div align="center">
<img src="/results/data.jpg" width="400" height="400">
</div>
<div align="center">Fig.1 Illustration of pre-processing for heart sound recordings.</div>

Common acoustic signal features are extracted through the openSMILE toolkit and are used for ML models.
   
``openSmile_raw": one-dimensional signal features from the raw recordings.

``openSmile_balanced": one-dimensional signal features from the balanced recordings.

Another modality involves image representation, where we adopted the continuous wavelet transform (CWT) for sound-image transformation used in DL models.
   
``image_CWT_raw": two-dimensional image features from the raw recordings.

``image_CWT_balanced": two-dimensional image features from the balanced recordings.

## Base Model

The Federated Learner: Binary=(RF, FNN, CNN)

gms: Gassian Mixture Models (List of GMMs)

The data to be decided (Acc, Se, Sp, UAR, UF1)

clfs: The binary classifiers;(List of Classifiers)

p_nodes: The proportion of the nodes; (List)


## Files

``Centralised_ensemble_learn.ipynb": We set up a set of baseline models based on data-centralised learning.

``Fed_ensemble_learn.ipynb": We present two medical institutions with three clients respectively, and perform homogeneous and heterogeneous stacked FL based on the local model under balanced data. 

## Remarks

In the code above, the scalar function is trained by full test data. We point out that this scalar is nothing but means and variances. One can equivalently train a scalar given the mean and variance of each node and their proportions in real processes.

## Contact

Please contact the authors of this work if you have any questions or comments.

## Cite As
Wanyong Qiu, Yifan Feng, Yuying Li, Yi Chang, Kun Qian*, Bin Hu∗, Yoshiharu Yamamoto and Bjoern W. Schuller, “Fed-MStacking: Heterogeneous Federated Learning with Stacking Misaligned Labels for Abnormal Heart Sound Detection”, IEEE JBHI, pp. 1-12, Submitted, October 2023.
