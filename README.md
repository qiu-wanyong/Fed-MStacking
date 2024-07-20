# Fed-MStacking
The Code of Fed-MStacking: Heterogeneous Federated Learning with Stacking Misaligned Labels for Abnormal Heart Sound  Detection

## Abstract
Ubiquitous sensing has been widely applied in smart healthcare, providing an opportunity for intelligent heart sound auscultation. However, smart devices contain sensitive information which concerns user privacy. To this end, federated learning (FL) has been adopted as an effective solution allowing decentralised learning without sharing data with each other, which helps to address data privacy in the Internet of Health Things (IoHT). Nevertheless, traditional FL requires the same architectural models to be trained across local clients and global servers, resulting in a lack of model heterogeneity and client personalisation. For medical institutions with private data clients, this study proposes Fed-MStacking, a heterogeneous FL framework, which incorporates the stacking ensemble learning strategy to support clients in building their own models. The secondary objective of this study is to address the scenario involving local clients with data characterised by inconsistent labelling. Particularly, the local client contains only one of the cases and the data cannot be shared within or outside the institution. In order to train a global multi-class classifier, we aggregate missing classes information from all clients at each institution and build meta-data, which then participates in an FL training step via a meta-learner. We apply the proposed framework to a multi-institutional heart sound database. The experiments utilise random forests (RFs), feedforward neural networks (FNNs), and convolutional neural networks (CNNs) as base classifiers. The results show that the heterogeneous stacking of the local models performs better compared to homogeneous stacking. 

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

Our work is ongoing. Based on this study, we are conducting follow-up research. Some of our new insights are below, welcome to exchange and share. 

*) Integrating dynamic data changes into the federated learning (FL) modelling process to promptly evolve the model is crucial in medical modelling. Dynamic FL, based on federated class-incremental and continual learning, will be a key research area for extending our work.

*) Combining swarm learning with FL has the potential to eliminate central servers during model training, allowing data and model updates to be exchanged directly between federated participants, further enhancing privacy protection in the FL system. 

## Cite As

Qiu W, Feng Y, Li Y, et al. Fed-MStacking: Heterogeneous Federated Learning with Stacking Misaligned Labels for Abnormal Heart Sound Detection[J]. IEEE Journal of Biomedical and Health Informatics, pp.1-12, 2024.
