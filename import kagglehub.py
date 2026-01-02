import pandas as pd
import os
from glob import glob

# مسیر فایل‌ها
data_path = "data/students-performance/raw data/"

# همه فایل‌های train
train_files = sorted(glob(os.path.join(data_path, "*_train.csv")))
# همه فایل‌های labels
label_files = sorted(glob(os.path.join(data_path, "*_labels.csv")))

# ترکیب train
train_list = [pd.read_csv(f) for f in train_files]
train_df = pd.concat(train_list, ignore_index=True)

# ترکیب labels
label_list = [pd.read_csv(f) for f in label_files]
labels_df = pd.concat(label_list, ignore_index=True)

print("Train shape:", train_df.shape)
print("Labels shape:", labels_df.shape)
