import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define the ANN model
class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Load the CSV dataset
# data = pd.read_csv('openSmile_6373/c.csv')

data = pd.read_csv('openSmile_resampled/overall.csv')

# data = pd.read_csv('openSmile_raw/overall.csv')

# Split the data into features (X) and labels (y)
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model and optimizer
input_size = X_train.shape[1]
model = ANNModel(input_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()

# Make predictions on the testing data
with torch.no_grad():
    y_pred_probs = model(X_test_tensor)
    y_pred = (y_pred_probs >= 0.5).squeeze().numpy()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
uf1 = (2 * tp) / (2 * tp + fp + fn)
uar = (sensitivity + specificity) / 2

classification_rep = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
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
