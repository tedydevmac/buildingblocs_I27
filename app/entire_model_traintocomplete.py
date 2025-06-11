import os
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.video import r3d_18
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
from tqdm import tqdm

class WLASLDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def read_video(self, frame_dir):
        frames = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
        video = []
        for f in frames:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            video.append(img)
        video = torch.stack(video)
        video = video.permute(1, 0, 2, 3)
        return video

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video = self.read_video(row['frame_dir'])
        label = row['label_idx']
        return video, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
])

df = pd.read_csv('metadata.csv')
classes = sorted(df['label'].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
df['label_idx'] = df['label'].map(class_to_idx)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_idx'])

train_dataset = WLASLDataset(train_df, transform=transform)
val_dataset = WLASLDataset(val_df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

class SignRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(SignRecognizer, self).__init__()
        self.backbone = r3d_18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def train_model(model, train_loader, val_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for videos, labels in tqdm(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {correct/total*100:.2f}%")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Validation Accuracy: {correct/total*100:.2f}%\n")

    return model

num_classes = len(classes)
model = SignRecognizer(num_classes=num_classes)
trained_model = train_model(model, train_loader, val_loader, epochs=10)

torch.save(trained_model.state_dict(), 'sign_model.pth')
