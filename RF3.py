import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, f1_score

# data = pd.read_csv('openSmile_resampled/overall.csv')

# data = pd.read_csv('openSmile_raw/f.csv')

data = pd.read_csv('openSmile_balanced/f.csv')

# Split the data into features (X) and labels (y)
X = data.drop('y', axis=1)
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators_custom = 50
max_depth_custom = 10
min_samples_split_custom = 5
criterion_custom = 'gini'

clf = RandomForestClassifier(n_estimators=n_estimators_custom,
                                   max_depth=max_depth_custom,
                                   min_samples_split=min_samples_split_custom,
                                   criterion=criterion_custom,
                                   random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
uf1 = (2 * tp) / (2 * tp + fp + fn)
uar = (sensitivity + specificity) / 2

# Calculate UAR and F1
# uar_macro = recall_score(y_test, y_pred, average='macro')
# uf1_macro = f1_score(y_test, y_pred, average='macro')

# Calculate Classification Report
classification_rep = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Set up the plot
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '46'
sns.set(font_scale=1.2)
sns.heatmap(normalized_conf_matrix, annot=True, cmap="Blues", fmt=".2f", xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Normalised Confusion Matrix')

plt.show()


print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Unweighted F1-Score (UF1):", uf1)
print("Unweighted Average Recall (UAR):", uar)
print("Classification Report:\n", classification_rep)
