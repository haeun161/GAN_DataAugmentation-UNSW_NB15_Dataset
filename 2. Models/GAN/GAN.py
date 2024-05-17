import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# GAN 모델 정의
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

# CSV 파일 읽기
file_path = 'GAN/data/unsw_train_secure.csv' #보안 관련 데이터를 제거한 UNSW 데이터

df = pd.read_csv(file_path)

#print(df.info())
# attack_cat 열을 기반으로 데이터를 분리
attack_categories = sorted(df['attack_cat'].unique())  # [0 1 2 3 6 7] or 다 사용 [1,2,3,4,5,6,7,8,9,10]

# Get counts for each attack category
attack_cat_counts = df['attack_cat'].value_counts()

# Update num_data_per_category with counts from attack_cat_counts
num_data_per_category = [60000 - attack_cat_counts.get(cat, 0) for cat in attack_categories]
    
num = 0
# 모든 attack_cat 값에 대해 반복
for target_attack_cat, target_num_data in zip(attack_categories, num_data_per_category):
    # attack_cat 열이 현재 target_attack_cat 값인 행을 추출
    df_target = df[df["attack_cat"] == target_attack_cat]

    # 데이터 정규화
    scaler = StandardScaler()
    data_array = df_target.to_numpy()
    data_array[:, :-1] = scaler.fit_transform(data_array[:, :-1])
    data_tensor = torch.FloatTensor(data_array)

    # 데이터 로더 설정
    batch_size = 64
    num_epochs = 150
    display_step = 25

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
    num = num+1
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

        if epoch % display_step == 0:
            print(f"Epoch [{epoch+25}/{num_epochs}] Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

    
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
        generated_data_list.append(generated_samples)            

# 2D 데이터를 Numpy 배열로 변환
generated_data_array = np.vstack(generated_data_list)
# Numpy 배열에서 Pandas DataFrame으로 변환
generated_data_df = pd.DataFrame(generated_data_array, columns=df.columns)  # Use the same columns as the original DataFrame

# 데이터 저장
generated_data_df.to_csv("/Data/2. GAN_Generated_Dataset/data1", index=False)  # Save the DataFrame to a CSV file
print("Generated data saved")
