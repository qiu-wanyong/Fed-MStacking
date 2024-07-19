import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_folder, labels_data, transform=None):
        self.image_folder = image_folder
        self.labels_data = labels_data
        self.transform = transform

    def __len__(self):
        return len(self.labels_data)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.image_folder, f'e{str(idx+1).zfill(5)}.png')
        img_name = os.path.join(self.image_folder, self.labels_data.iloc[idx, 0])

        image = Image.open(img_name).convert('RGB')
        label = self.labels_data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# raw data
# image_folder = 'image_cwt_raw/overall'
# labels_file = 'image_cwt_raw/overall_labels.csv'

image_folder = 'CWT_data_resampled/overall'
labels_file = 'CWT_data_resampled/overall_labels.csv'

# balanced data
# image_folder = 'CWT_data_resampled/training-e1'
# labels_file = 'CWT_data_resampled/training-e/e_labels.csv'

# Load and split the data
labels_data = pd.read_csv(labels_file)
train_data, test_data = train_test_split(labels_data, test_size=0.3, random_state=42)

train_dataset = CustomDataset(image_folder, train_data, transform=transform)
test_dataset = CustomDataset(image_folder, test_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the CNN model and optimizer
model = CNNModel()
criterion = nn.BCELoss()  # Binary Cross-Entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_probs = []
    y_true = []

    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        y_pred_probs.extend(outputs)
        y_true.extend(batch_y)

y_pred_probs = torch.cat(y_pred_probs)
y_pred = (y_pred_probs >= 0.5).float()
y_true = torch.tensor(y_true)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
uf1 = (2 * tp) / (2 * tp + fp + fn)
uar = (sensitivity + specificity) / 2

# Calculate the normalized confusion matrix
normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Print metrics
classification_rep = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])
normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Unweighted F1-Score (UF1):", uf1)
print("Unweighted Average Recall (UAR):", uar)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '46'
sns.set(font_scale=1.2)
sns.heatmap(normalized_conf_matrix, annot=True, cmap="Blues", fmt=".2f", xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Normalised Confusion Matrix')
print("Classification Report:\n", classification_rep)

plt.show()

