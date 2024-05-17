import pandas as pd
import numpy as np

# 원본 데이터 로드
file_path = 'GAN/data/unsw_train.csv'
original_data = pd.read_csv(file_path)

# 생성된 데이터 로드
file_path = 'GAN/Generated_Data/lr/16_50_0.0001.csv'
generated_data = pd.read_csv(file_path)

'''
#데이터셋 개수 체크
attack_cat_counts1 = original_data['attack_cat'].value_counts()
print("기존 training 데이터 label별 개수:")
print(attack_cat_counts1)
'''

# 두 데이터프레임을 수직으로 합치기
combined_data = pd.concat([original_data, generated_data], axis=0)

# Count the occurrences of each attack category
attack_cat_counts2 = combined_data['attack_cat'].value_counts()
print("합친 데이터 label별 개수:")
print(attack_cat_counts2)

# 데이터를 무작위로 섞기
combined_data = combined_data.sample(frac=1, random_state=42)  # random_state를 지정하여 재현성을 보장할 수 있습니다.

# 결과 데이터프레임을 저장
combined_data.to_csv('/Data/3. GAN_Augmented Dataset/dataset1', index=False)