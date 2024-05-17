import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from collections import Counter
import tracemalloc
import time
from datetime import datetime

# Load your train dataset
file_path = '/Data/4. Secured Dataset/unsw_train_secure.csv'

# 리스트를 저장할 전역 변수
smote_elapsed_times = []
gan_elapsed_times = []
smote_memory_usage = []
gan_memory_usage = []


def start_memory_profiling():
    tracemalloc.start()

def stop_memory_profiling():
    current, peak = tracemalloc.get_traced_memory()
    smote_memory_usage.append((current, peak))
    tracemalloc.stop()

def stop_memory_profiling1():
    current, peak = tracemalloc.get_traced_memory()
    gan_memory_usage.append((current, peak))
    tracemalloc.stop()

###############SMOTE로 데이터 생성하기###############
def oversample_with_smote(data, target_count, k_neighbors=4, random_state=42):
    X = data.drop("attack_cat", axis=1)  # 특성 데이터
    y = data["attack_cat"]  # 라벨 데이터

    # 각 클래스의 현재 개수를 계산
    current_count = Counter(y)

    # 각 클래스에 대한 oversampling 비율을 계산
    sampling_strategy = {label: target_count for label in current_count.keys()}

    # SMOTE를 초기화하고 적용
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Oversampling 이후의 클래스 분포 확인
    print("Class distribution after SMOTE:", Counter(y_resampled))

    # X_resampled와 y_resampled를 원본 데이터프레임 형식으로 변환
    resampled_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame({"attack_cat": y_resampled})], axis=1)

    # Check the balanced count for each attack category
    balanced_attack_cat_counts = resampled_data['attack_cat'].value_counts()
    print("SMOTE로 ovesampling된 데이터 개수:")
    print(balanced_attack_cat_counts)

    return resampled_data

def smote(fraction):
    train_data = pd.read_csv(file_path)
    train_data = train_data.dropna()

    # Sample 10% of the entire training dataset
    sampled_data = train_data.sample(frac=fraction, random_state=42)

    # 각 attack_Cat 라벨에 대해 60,000개의 데이터가 생성되도록 oversample_with_smote 함수 호출
    oversampled_data = oversample_with_smote(sampled_data, target_count=60000, random_state=42)

    # Save the oversampled data to a CSV file
    oversampled_data.to_csv(f'GAN/Generated_Data_secure/smote1/smote_{fraction}.csv', index=False)
    print("Saved SMOTE oversampled data")


###############GAN으로 데이터 생성하기###############
def gan(fraction):
     # Define the Generator
    class Generator(nn.Module):
        def __init__(self, input_size, output_size):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, output_size),
                nn.Tanh()
            )
        
        def forward(self, z):
            return self.model(z)

    # Define the Discriminator
    class Discriminator(nn.Module):
        def __init__(self, input_size):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    generated_data_list = []

    # train 데이터셋 가져오기
    df = pd.read_csv(file_path)
    df =df.dropna()
    # print(df.info())

    # Sample 10% of the entire training dataset
    sampled_data = df.sample(frac=fraction, random_state=42)

    # attack_cat 열을 기반으로 데이터를 분리
    attack_categories = sorted(sampled_data['attack_cat'].unique())  # [0 1 2 3 6 7] or 다 사용 [0,1,2,3,4,5,6,7,8,9]

    # Get counts for each attack category
    attack_cat_counts = sampled_data['attack_cat'].value_counts()

    # Update num_data_per_category with counts from attack_cat_counts
    num_data_per_category = [60000 - attack_cat_counts.get(cat, 0) for cat in attack_categories]
    
    num = 0
    # 모든 attack_cat 값에 대해 반복
    for target_attack_cat, target_num_data in zip(attack_categories, num_data_per_category):
        # attack_cat 열이 현재 target_attack_cat 값인 행을 추출
        target = sampled_data[sampled_data["attack_cat"] == target_attack_cat]
        
        # 데이터 정규화
        scaler = StandardScaler()
        data_array = target.to_numpy()
        data_array[:, :-1] = scaler.fit_transform(data_array[:, :-1])
        data_tensor = torch.FloatTensor(data_array)

        # 데이터 로더 설정
        batch_size = 64
        num_epochs = 150
        #display_step = 50

        dataset = TensorDataset(data_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # GAN 모델 초기화, 훈련 및 생성
        input_size = 10  # 잠재 공간의 차원
        output_size = data_array.shape[1]  # 데이터 프레임의 열 수

        generator = Generator(input_size, output_size)
        discriminator = Discriminator(output_size)

        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer_G = optim.Adam(generator.parameters(), lr=0.0008)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0008)

        # Training loop
        num += 1
        print(num)
        for epoch in range(num_epochs):        
            for real_data in data_loader:
                real_data = real_data[0]  # Extract the actual data from the DataLoader tuple
                batch_size = real_data.shape[0]  # Get the batch size from the actual data

                # Labels (adjust size to match batch_size)
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # Generate fake data using the generator
                z = torch.randn(batch_size, input_size)
                generated_data = generator(z)

                # Discriminator forward and backward
                optimizer_D.zero_grad()
                d_real = discriminator(real_data)
                d_fake = discriminator(generated_data.detach())
                loss_real = criterion(d_real, real_labels)
                loss_fake = criterion(d_fake, fake_labels)
                loss_d = loss_real + loss_fake
                loss_d.backward()
                optimizer_D.step()

                # Generator forward and backward
                optimizer_G.zero_grad()
                d_fake = discriminator(generated_data)
                loss_g = criterion(d_fake, real_labels)
                loss_g.backward()
                optimizer_G.step()

        # 학습 후 생성된 데이터를 저장
        with torch.no_grad():
            z = torch.randn(target_num_data, input_size)
            generated_samples = generator(z).numpy()

            # 스케일링된 열의 복원
            # 원본 스케일러를 사용하여 스케일을 복원합니다
            scaler.fit(data_array[:, :-1])  # Reinitialize the scaler
            generated_samples[:, :-1] = scaler.inverse_transform(generated_samples[:, :-1])

            # Update the 'attack_cat' values to the target_attack_cat
            generated_samples[:, -1] = target_attack_cat

        # 생성된 샘플을 리스트에 추가합니다
            if len(generated_data_list) == 0:
                generated_data_list = generated_samples
            else:
                generated_data_list = np.vstack((generated_data_list, generated_samples))

                
    # 2D 데이터를 Numpy 배열로 변환
    generated_data_array = np.vstack(generated_data_list)
    # Numpy 배열에서 Pandas DataFrame으로 변환
    generated_data_df = pd.DataFrame(generated_data_array, columns=sampled_data.columns)  # Use the same columns as the original DataFrame



    ###############생성된 데이터 합치기###############
    # 두 데이터프레임을 수직으로 합치기
    combined_data = pd.concat([sampled_data, generated_data_df], axis=0)
    # Count the occurrences of each attack category
    attack_cat_counts2 = combined_data['attack_cat'].value_counts()
    #print("합친 데이터 label별 개수:")
    #print(attack_cat_counts2)

    # Save the oversampled data to a CSV file
    combined_data.to_csv(f'GAN/Generated_Data_secure/gan/gan_{fraction}.csv', index=False)
    print("saved GAN oversampled data")

if __name__ == "__main__":
    for fraction in np.arange(0.1, 1.1, 0.1):
        
        start_memory_profiling()
        start_time1 = time.time()
        gan(fraction)
        end_time1 = time.time()
        stop_memory_profiling1()
        elapsed_time1 = end_time1 - start_time1
        gan_elapsed_times.append(elapsed_time1)


    print("SMOTE Elapsed Times:", smote_elapsed_times)
    print("GAN Elapsed Times:", gan_elapsed_times)
    print("SMOTE Memory Usage:", smote_memory_usage)
    print("GAN Memory Usage:", gan_memory_usage)

#--------------------------
#지연도랑 메모리 사용률 측정
#불균형 데이터셋부터 시작해서 over-sampling하는 과정 시간 측정
#데이터 샘플링 사이즈 10%씩 잘라서 각각 측정