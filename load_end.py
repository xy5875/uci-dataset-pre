import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(561, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data = torch.tensor(pd.read_csv(data_file).values, dtype=torch.float32)
        self.labels = torch.tensor(pd.read_csv(label_file).values.flatten(), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def make_dataloader(data_path, subject, batch_size=10, num_workers=0):
    train_data_file = os.path.join(data_path, f"subject_{subject}_train.csv")
    train_labels_file = os.path.join(data_path, f"subject_{subject}_train_labels.csv")
    test_data_file = os.path.join(data_path, f"subject_{subject}_test.csv")
    test_labels_file = os.path.join(data_path, f"subject_{subject}_test_labels.csv")

    if not os.path.exists(train_data_file) or not os.path.exists(train_labels_file) or \
       not os.path.exists(test_data_file) or not os.path.exists(test_labels_file):
        return None

    train_dataset = CustomDataset(train_data_file, train_labels_file)
    test_dataset = CustomDataset(test_data_file, test_labels_file)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, test_dataloader



def train(net, dataloader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
    #读取每个subject的dataloader
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss = {running_loss/len(dataloader)}, Accuracy = {accuracy}%")



# 使用示例
net = Net()
data_path = 'E:/OPERATION/data'  # 替换为您的数据路径
train_dataloader, test_dataloader = make_dataloader(data_path, subject=1)
train(net, train_dataloader)
