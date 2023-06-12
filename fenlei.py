import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义数据集文件夹路径
dataset_folder = r'E:\OPERATION\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset'  # 替换为您的数据集文件夹路径

#创建存储数据的文件夹
if not os.path.exists("data"):
    os.makedirs("data")


# 读取受试者数据、标签和受试者信息
train_data_file = os.path.join(dataset_folder, "test/X_test.txt")
train_labels_file = os.path.join(dataset_folder, "test/y_test.txt")
train_subjects_file = os.path.join(dataset_folder, "test/subject_test.txt")

train_data = pd.read_csv(train_data_file, delim_whitespace=True, header=None, names=list(range(1, 562)))
train_labels = pd.read_csv(train_labels_file, delim_whitespace=True, header=None, names=["Label"])
train_subjects = pd.read_csv(train_subjects_file, delim_whitespace=True, header=None, names=["Subject"])
#train_subjects = train_subjects.reset_index(drop=True)

# 根据受试者类别进行数据划分
unique_subjects = set(train_subjects["Subject"])  # 获取唯一的受试者类别
for subject in unique_subjects:
    subject_indices = train_subjects[train_subjects["Subject"] == subject].index  # 获取当前受试者的数据索引
    subject_data = train_data.loc[subject_indices]  # 获取当前受试者的数据
    subject_labels = train_labels.loc[subject_indices]  # 获取当前受试者的标签
    subject_name = f"subject_{subject}"  # 根据受试者类别生成命名

    # 保存受试者数据为本地文件
    file_name = f"{subject_name}_data.csv"
    file_path = os.path.join("data", file_name)
    subject_data.to_csv(file_path, index=False)

    # 对当前受试者数据进行三七分划分
    X_train, X_test, y_train, y_test = train_test_split(
        subject_data, subject_labels, test_size=0.3, random_state=42
    )

    # 保存训练数据集为本地文件
    train_file_name = f"{subject_name}_train.csv"
    train_file_path = os.path.join("data", train_file_name)
    X_train.to_csv(train_file_path, index=False)
    train_labels_file_path = os.path.join("data", f"{subject_name}_train_labels.csv")
    y_train.to_csv(train_labels_file_path, index=False)

    # 保存测试数据集为本地文件
    test_file_name = f"{subject_name}_test.csv"
    test_file_path = os.path.join("data", test_file_name)
    X_test.to_csv(test_file_path, index=False)
    test_labels_file_path = os.path.join("data", f"{subject_name}_test_labels.csv")
    y_test.to_csv(test_labels_file_path, index=False)

# 创建info.json文件
info = {}
for subject in unique_subjects:
    subject_name = f"subject_{subject}"
    info[subject_name] = {
        "delay": 0.1  # 替换为适当的模拟延时
    }

info_file_path = os.path.join("data", "info.json")
with open(info_file_path, "w") as f:
    json.dump(info, f, indent=4)
