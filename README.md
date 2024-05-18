# GAN-UNSW-NB15-Dataset
Compartion of Models(SMOTE, GAN) for Data Augmentation with UNSW-NB15 Dataset
- <b>Proposed Model</b>: GAN Data Augmentation + Security-related Feature elimination
- <b>Compare Model</b>: 1. Data Augmentation x 2. SMOTE 3.GAN

# About Dataset
<b> UNSW-NB15 </b>
UNSW-NB15 is a network intrusion dataset. It contains nine different attacks, includes DoS, worms, Backdoors, and Fuzzers. 
The dataset contains raw network packets. The number of records in the training set is 175,341 records and the testing set is 82,332 records from the different types, attack and normal.
<b>Link</b>: https://paperswithcode.com/dataset/unsw-nb15

<b> Train set & Test set </b>
![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/223f5783-e7bd-4606-b20a-12ffc78c474a)

* In this experiment, we used Trainset for Data Augmentation

# Backgroud (Data Analysis -> Proposed Solution)
<b> (1) Data Analysis </b>
In this dataset, the number of Attack Category datasets (Backdoor, Analysis, ShellCode, Worms) is significantly smaller than others. When the number of instances for each category is highly imbalanced during classification, several problems can arise.

Such as..
- <b> 1. Model Bias: </b>  The model may become biased towards the majority class, leading to poor performance on minority classes.
- <b> 2. Poor Generalization: </b>  The model might not learn the characteristics of the minority classes well, resulting in poor generalization when making predictions on new data.
- <b> 3. Skewed Metrics: </b> Evaluation metrics such as accuracy may be misleading, as a high accuracy can be achieved by simply predicting the majority class.
- <b> 4. Overfitting: </b> The model may overfit the majority class data, capturing noise instead of the underlying patterns.

<b> (2) Proposed Solution </b>
 1) Make balanced data by augmenting data
 2) To enhance security, remove some security-related features during training

# Model Selction
*Notation: the linked Notion Page is written in Korean.
1. Research on Gernerative AI for Data Augmentation -> select Gernerative AI Model
   - https://button-breeze-d77.notion.site/AI-Generative-AI-data-augmentation-c6391a3b082f403591e913ae3cd94661?pvs=4
2. Related Work
   - Network Intrusion Detection Based on Supervised Adversarial Variational Auto-Encoder With Regularization


# Experiments and analysis
<b> 1. Data Augmentation x -> Train </b>
   Evaluation Metric: Accuracy
![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/690edcf9-4a3b-43f8-a46c-73ec469ad108)
 - Data: ['SMOTE oversampled Data', 'GAN oversampled Data']
 - Training Accuracy: [99.9045918367347, 100.0]
 - Test Accuracy: [85.42678571428571, 99.995290349927]

<b> 2. GAN & SMOTE Data Augmentation o -> Train </b>
- <b> Evaluation Metric</b>: Accuracy, Memory Usage, Elapsed Time 
![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/bd0c58ea-ab87-4cc0-96ef-a986b8776883)

  * Working on solving Overfitting Problem

<b> 3. Security-related Feature elimination -> GAN & SMOTE Data Augmentation o -> Train </b>
- <b> Evaluation Metric</b>: Accuracy, Memory Usage, Elapsed Time 
  ![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/6e3b7ffe-c6cb-4417-8cef-748f3f54023e)

  * Working on solving Overfitting Problem



# Usage
### Dataset:
used preprocessed UNSW-NB15 Dataset as datset.csv 

### Run
run `python main.py`
