import pandas as pd
from imblearn.over_sampling import SMOTE

# Load your dataset
file_path = 'GAN/data/unsw_train.csv'
data = pd.read_csv(file_path)

# Count the occurrences of each attack category
attack_cat_counts = data['attack_cat'].value_counts()
print(attack_cat_counts)

# Find the maximum count of attack categories
max_count = attack_cat_counts.max()
print(max_count)

# Instantiate SMOTE
smote = SMOTE(sampling_strategy={i: max_count for i in attack_cat_counts.index})

# Oversample the dataset
X = data.drop(columns=['attack_cat'])
y = data['attack_cat']
X_resampled, y_resampled = smote.fit_resample(X, y)

# Create a new balanced DataFrame
balanced_data = pd.concat([X_resampled, y_resampled], axis=1)

# Check the balanced count for each attack category
balanced_attack_cat_counts = balanced_data['attack_cat'].value_counts()
print(balanced_attack_cat_counts)

# Save the oversampled data to a CSV file
oversampled_file_path = 'GAN/data/unsw_oversampled_data.csv'
balanced_data.to_csv(oversampled_file_path, index=False)

print(f'Oversampled data saved to {oversampled_file_path}')



data = pd.read_csv('/data/unsw_oversampled_data.csv')
print(data['attack_cat'].value_counts())
