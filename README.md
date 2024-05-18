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
![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/557c1bc5-e188-43e0-82cd-ae5045ab5540/5dbbe20b-1c93-4fe1-8fb2-9953c4e102e2/Untitled.png)
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
- 1) Make balanced data by augmenting data
- 2) To enhance security, remove some security-related features during training

# Model Selction
*Notation: the linked Notion Page is written in Korean.
- 1. Research on Gernerative AI for Data Augmentation -> select Gernerative AI Model
   https://button-breeze-d77.notion.site/AI-Generative-AI-data-augmentation-c6391a3b082f403591e913ae3cd94661?pvs=4
- 2. Related Work
   - Network Intrusion Detection Based on Supervised Adversarial Variational Auto-Encoder With Regularization


# Experiments and analysis
<b> 1. Data Augmentation x -> Train </b>
   Evaluation Metric: Accuracy
![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/690edcf9-4a3b-43f8-a46c-73ec469ad108)
 - Data: ['SMOTE oversampled Data', 'GAN oversampled Data']
 - Training Accuracy: [99.9045918367347, 100.0]
 - Test Accuracy: [85.42678571428571, 99.995290349927]

<b> 2. GAN & SMOTE Data Augmentation o -> Train </b>
   <b> Evaluation Metric: Accuracy </b>
   ![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/e959677f-f4ac-4631-9f7f-2ee750000925)
   - need to solve Overfitting Problem
   <b> Evaluation Metric: Memory Usage </b>
   ![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/4e4d061d-8356-424f-9b1c-ef9f8ceffae8)
   <b> Evaluation Metric: Elapsed Time </b>
  ![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/3282fefd-727a-4b62-9cd6-02c32ae4d17e)


<b> 3. Security-related Feature elimination -> GAN & SMOTE Data Augmentation o -> Train </b>
   <b> Evaluation Metric: Accuracy </b>
   ![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/5a270f33-6fcd-4a2c-a5a6-031008e0177f)
   <b> Evaluation Metric: Memory Usage </b>
   ![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/1a0290b7-8f64-496e-977a-8229888398c3)
   <b> Evaluation Metric: Elapsed Time </b>
   ![image](https://github.com/haeun161/GAN-UNSW-NB15-Dataset/assets/80445078/6728676d-cef6-44f3-8bd1-f72ca1f9931c)



# Usage
### Dataset:
used preprocessed UNSW-NB15 Dataset as datset.csv 

### Run
run `python main.py`
