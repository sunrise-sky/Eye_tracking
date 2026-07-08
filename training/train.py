import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

DATA_DIR = r'D:\EyeTracking\LPW\lpw_extracted_2k'
MODEL_SAVE_NAME = 'pupil_gap.pth'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0003


class LPWDataset224(Dataset):
    def __init__(self, root_dir):
        self.img_dir = os.path.join(root_dir, "images")
        with open(os.path.join(root_dir, "labels.txt"), 'r') as f:
            self.lines = f.readlines()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip().split()
        img_name, nx, ny = line[0], float(line[1]), float(line[2])
        img = cv2.imread(os.path.join(self.img_dir, img_name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            nx, ny = 0.5, 0.5
        img = self.clahe.apply(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        label_tensor = torch.tensor([nx, ny], dtype=torch.float32)
        return img_tensor, label_tensor


class PupilGAP(nn.Module):
    def __init__(self):
        super(PupilGAP, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AvgPool2d(kernel_size=14)
        self.fc1 = nn.Linear(64, 32)
        self.relu5 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.relu5(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")

    dataset = LPWDataset224(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PupilGAP().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs[:, 0], lbls[:, 0]) + 2.0 * criterion(outputs[:, 1], lbls[:, 1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.6f}")

    # 保存损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS+1), losses, 'b-o', markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PupilGAP Training Loss Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    print("损失曲线已保存为 training_loss.png")

    torch.save(model.state_dict(), MODEL_SAVE_NAME)
    print(f"模型已保存为: {MODEL_SAVE_NAME}")


if __name__ == "__main__":
    train()
