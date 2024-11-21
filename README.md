# AERL & AERL-FS
This repo contains the working code for AERL & AERL-FS agorithms.

Requirements -
1. pip install shap
2. pip install xgboost


**Abstract**-
Supervised feature selection aims to identify a subset of rele-
vant features from high-dimensional data to enhance model accuracy and
efficiency. Traditional feature selection techniques often produce subop-
timal feature subsets due to limitations in capturing non-linear interac-
tions and complex feature relationships, poor algorithmic stability, and
susceptibility to the curse of dimensionality. To tackle these challenges,
We proposed two algorithms for feature selection in supervised learn-
ing: the Autoencoder with Enhanced Reconstruction Loss (AERL) and
an advanced version, AERL with Feature Scoring (AERL-FS). AERL
employs an autoencoder with a custom loss function to address the com-
plexities of high-dimensional data more comprehensively than traditional
methods. AERL-FS builds on this by adding a feature importance mech-
anism for effective feature selection. Comparative evaluations show that
our proposed methods outperform state-of-the-art algorithms in terms
of model accuracy.
