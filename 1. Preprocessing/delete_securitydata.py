#보안 관련 데이터가 들어간 데이터 중 일부 NaN 값으로 대체

import pandas as pd
import numpy as np

# 데이터 로드
data = pd.read_csv('GAN/data/unsw_train.csv')

# 랜덤 시드 설정
np.random.seed(42)

# 랜덤하게 선택할 열(label) 목록
columns_to_nan = ['service', 'state'] #'service', 

# 데이터의 절반만큼을 랜덤하게 NaN으로 설정
for column in columns_to_nan:
    # 랜덤하게 선택된 행 인덱스
    nan_indices = np.random.choice(data.index, len(data) // 2, replace=False)
    
    # 선택된 행의 해당 열을 NaN으로 설정
    data.loc[nan_indices, column] = np.nan

# 결과 확인
print(data.head())

# NaN으로 만든 데이터프레임을 새로운 CSV 파일로 저장
data.to_csv('GAN/data/unsw_train_secure.csv', index=False)
