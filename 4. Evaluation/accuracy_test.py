import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv('GAN/data/unsw_train.csv')
# 데이터 세트 정의
gan_data1 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_0.1.csv') 
gan_data2 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_0.2.csv')
gan_data3 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_0.3.csv')
gan_data4 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_0.4.csv')
gan_data5 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_0.5.csv')
gan_data6 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_0.6.csv')
gan_data7 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_0.7.csv')
gan_data8 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_0.8.csv')
gan_data9 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_0.9.csv')
gan_data10 = pd.read_csv('GAN/Generated_Data_secure/gan1/gan_1.0.csv')

smote_data1 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_0.1.csv')
smote_data2 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_0.2.csv')
smote_data3 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_0.3.csv')
smote_data4 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_0.4.csv')
smote_data5 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_0.5.csv')
smote_data6 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_0.6.csv')
smote_data7 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_0.7.csv')
smote_data8 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_0.8.csv')
smote_data9 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_0.9.csv')
smote_data10 = pd.read_csv('GAN/Generated_Data_secure/smote1/smote_1.0.csv')

datasets_gan = [gan_data1, gan_data2, gan_data3, gan_data4, gan_data5, gan_data6, gan_data7, gan_data8, gan_data9, gan_data10]
datasets_smote = [smote_data1, smote_data2, smote_data3, smote_data4, smote_data5, smote_data6, smote_data7, smote_data8, smote_data9, smote_data10]
data_names = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

# 모델 초기화
model = DecisionTreeClassifier()

# 결과 저장을 위한 딕셔너리
results_gan = {
    "Data": [],
    "Training Accuracy": [],
    "Test Accuracy": []
}

results_smote = {
    "Data": [],
    "Training Accuracy": [],
    "Test Accuracy": []
}

# 데이터 세트별로 반복
for gan_dataset, smote_dataset, data_name in zip(datasets_gan, datasets_smote, data_names):
    gan_data = gan_dataset
    smote_data = smote_dataset
    
    gan_data = gan_data.sample(frac=1, random_state=42)  # 데이터 셔플
    smote_data = smote_data.sample(frac=1, random_state=42)  # 데이터 셔플
    
    # 데이터 전처리 및 분할 - GAN
    X_gan = gan_data.drop(columns=['attack_cat'])
    X_gan = np.array(X_gan)
    y_gan = gan_data['attack_cat']
    X_train_gan, X_test_gan, y_train_gan, y_test_gan = train_test_split(X_gan, y_gan, test_size=0.3, random_state=42)
    
    # 데이터 스케일링 - GAN
    scaler_gan = StandardScaler().fit(X_train_gan)
    X_train_gan = scaler_gan.transform(X_train_gan)
    X_test_gan = scaler_gan.transform(X_test_gan)
    
    # 모델 학습 - GAN
    model.fit(X_train_gan, y_train_gan)
    
    # 모델 성능 측정 - GAN
    train_accuracy_gan = accuracy_score(y_train_gan, model.predict(X_train_gan))
    test_accuracy_gan = accuracy_score(y_test_gan, model.predict(X_test_gan))
    
    # 결과 저장 - GAN
    results_gan["Data"].append(data_name)
    results_gan["Training Accuracy"].append(train_accuracy_gan * 100)
    results_gan["Test Accuracy"].append(test_accuracy_gan * 100)
    
    # 데이터 전처리 및 분할 - SMOTE
    X_smote = smote_data.drop(columns=['attack_cat'])
    X_smote = np.array(X_smote)
    y_smote = smote_data['attack_cat']
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)
    
    # 데이터 스케일링 - SMOTE
    scaler_smote = StandardScaler().fit(X_train_smote)
    X_train_smote = scaler_smote.transform(X_train_smote)
    X_test_smote = scaler_smote.transform(X_test_smote)
    
    # 모델 학습 - SMOTE
    model.fit(X_train_smote, y_train_smote)
    
    # 모델 성능 측정 - SMOTE
    train_accuracy_smote = accuracy_score(y_train_smote, model.predict(X_train_smote))
    test_accuracy_smote = accuracy_score(y_test_smote, model.predict(X_test_smote))
    
    # 결과 저장 - SMOTE
    results_smote["Data"].append(data_name)
    results_smote["Training Accuracy"].append(train_accuracy_smote * 100)
    results_smote["Test Accuracy"].append(test_accuracy_smote * 100)

# 그래프 생성
plt.figure(figsize=(15, 8))

# Combine GAN and SMOTE results in a single graph
plt.subplot(1, 2, 1)
sns.lineplot(x="Data", y="Test Accuracy", data=results_gan, label="GAN")
sns.lineplot(x="Data", y="Test Accuracy", data=results_smote, label="SMOTE")
plt.ylim(50, 100)  # Set the y-axis limits to be between 50% and 100%
plt.ylabel("Test Accuracy (%)")
plt.title("GAN and SMOTE Accuracy")

# ... (remaining code remains unchanged)

# 출력
plt.tight_layout()
plt.show()

# 출력 결과
print("GAN Results:")
print(pd.DataFrame(results_gan))

print("\nSMOTE Results:")
print(pd.DataFrame(results_smote))