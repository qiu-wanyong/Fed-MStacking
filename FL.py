import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


#  同构 RF_blanced
RF_homer_FL = []
#  异构 FL_blanced
RF_heterogeneity = []
#  Full RF
RF_raw = []
RF_blanced = []

#  同构 ANN_blanced
ANN_homer_FL = []
#  异构 FL_blanced
ANN_heterogeneity = []
#  Full RF
ANN_raw = []
ANN_blanced = []

#  同构 CNN_blanced
CNN_homer_FL = []
#  异构 FL_blanced
NN_heterogeneity = []
#  Full RF
CNN_raw = []
CNN_blanced = []

data_raw = pd.read_csv('openSmile_raw/overall.csv')  # Normal + VD + CAD
data_raw_VD = pd.read_csv('openSmile_raw/a.csv')  # VD
data_raw_CAD1 = pd.read_csv('openSmile_raw/b.csv')  # CAD
data_raw_CAD2 = pd.read_csv('openSmile_raw/e.csv')  # CAD
data_raw_CAD = pd.read_csv('openSmile_raw/be.csv')  # CAD

data_banced = pd.read_csv('openSmile_balanced/overall.csv')  # Normal + VD + CAD
data_banced_VD = pd.read_csv('openSmile_balanced/a.csv')  # VD
data_banced_CAD1 = pd.read_csv('openSmile_balanced/b.csv')  # CAD
data_banced_CAD2 = pd.read_csv('openSmile_balanced/e.csv')  # CAD
data_banced_CAD = pd.read_csv('openSmile_balanced/be.csv')  # CAD

# Calculate the proportion of positive samples (0) and negative samples (1)
positive_raw = (data_raw['y'] == 0).sum()  # Normal
negative_raw_VD = (data_raw_VD['y'] == 1).sum()  # VD
negative_raw_CAD1 = (data_raw_CAD1['y'] == 1).sum()  # CAD1
negative_raw_CAD2 = (data_raw_CAD2['y'] == 1).sum()  # CAD2
# negative_raw_CAD = negative_raw_CAD1 + negative_raw_CAD2  # CAD
negative_raw_CAD = (data_raw_CAD['y'] == 1).sum()  # CAD2
print("positive_raw:", positive_raw)
print("negative_raw_VD:", negative_raw_VD)
print("negative_raw_CAD:", negative_raw_CAD)

positive_banced = (data_banced['y'] == 0).sum()  # Normal
negative_banced_VD = (data_banced_VD['y'] == 1).sum()  # VD
negative_banced_CAD1 = (data_banced_CAD1['y'] == 1).sum()  # CAD1
negative_banced_CAD2 = (data_banced_CAD2['y'] == 1).sum()  # CAD2
# negative_banced_CAD = negative_banced_CAD1 + negative_banced_CAD2 # CAD
negative_banced_CAD = (data_banced_CAD['y'] == 1).sum()  # CAD2
print("positive_banced:", positive_banced)
print("negative_banced_VD:", negative_banced_VD)
print("negative_banced_CAD:", negative_banced_CAD)

total_samples_raw = len(data_raw)
total_samples_blanced = len(data_banced)
print("total_samples_raw:", total_samples_raw)
print("total_samples_blanced:", total_samples_blanced)

# Calculate the proportions
p_negative_raw_VD = negative_raw_VD / total_samples_raw  # VD
p_negative_raw_CAD = negative_raw_CAD / total_samples_raw  # CAD

p_negative_banced_VD = negative_banced_VD / total_samples_raw  # VD
p_negative_banced_CAD = negative_banced_CAD / total_samples_raw  # CAD

p_normal = positive_raw / total_samples_raw
p_normal_blanced = positive_banced / total_samples_blanced

# Store the proportions in p_labels and p_balanced
p_labels = [p_normal, p_negative_raw_VD, p_negative_raw_CAD]
p_balanced = [p_normal_blanced, p_negative_banced_VD, p_negative_banced_CAD]

print("p_raw:", p_labels)
print("p_balanced:", p_balanced)

# p_balanced = p_balanced[:len(recalls)]

# Set up the figure and axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
# width = 0.4

# fig, ax = plt.subplots()
width = 0.25
x1 = np.arange(3)


# Plot the data
'''
# raw data
ax.bar(x1, np.array(recalls), width, alpha=0.7, label='Ensembled (unbalanced)', color='maroon')
ax.bar(x1, np.array(accs_full), width, alpha=0.7, label='Full (unbalanced)', color='deepskyblue')
ax.bar(x1 + width, np.array(accs_b), width, alpha=0.7, label='Ensembled (balanced)', color='r')
ax.bar(x1 + width, np.array(recalls_full_b), width, alpha=0.7, label='Full (balanced)', color='b')
plt.plot(x1, p_labels, color='y', linewidth=5.0, linestyle= '--',label='Sample Proportion(unbalanced)')
plt.plot(x1 + width, p_balanced, color='lightsalmon', linewidth=5.0, linestyle= '-', label='Sample Proportion (balanced)')
'''
'''
# RF Precision
ax.bar(x1, np.array(P_RF_raw), width, alpha=0.7, label='Full-unbalanced (RF)', color='maroon')
ax.bar(x1 + width, np.array(P_RF_blanced), width, alpha=0.7, label='Full-balanced (RF)', color='r')
ax.bar(x1, np.array(P_RF_homer_FL), width, alpha=0.7, label='Homogeneity (RF)', color='deepskyblue')
ax.bar(x1 + width, np.array(P_RF_heterogeneity), width, alpha=0.7, label='Heterogeneity', color='b')

plt.plot(x1, p_labels, color='y', linewidth=5.0, linestyle = '--', label='Sample Proportion(unbalanced)')
plt.plot(x1 + width, p_balanced, color='lightsalmon', linewidth=5.0, linestyle= '-', label='Sample Proportion (balanced)')
'''
'''
# RF Se
ax.bar(x1, np.array(Se_RF_raw), width, alpha=0.7, label='Full-unbalanced (RF)', color='maroon')
ax.bar(x1 + width, np.array(Se_RF_blanced), width, alpha=0.7, label='Full-balanced (RF)', color='r')
ax.bar(x1, np.array(Se_RF_homer_FL), width, alpha=0.7, label='Homogeneity (RF)', color='deepskyblue')
ax.bar(x1 + width, np.array(Se_RF_heterogeneity), width, alpha=0.7, label='Heterogeneity', color='b')

plt.plot(x1, p_labels, color='y', linewidth=5.0, linestyle = '--', label='Sample Proportion(unbalanced)')
plt.plot(x1 + width, p_balanced, color='lightsalmon', linewidth=5.0, linestyle= '-', label='Sample Proportion (balanced)')
'''

'''
# ANN Precision
ax.bar(x1, np.array(P_ANN_raw), width, alpha=0.7, label='Full-unbalanced (RF)', color='maroon')
ax.bar(x1 + width, np.array(P_ANN_blanced), width, alpha=0.7, label='Full-balanced (RF)', color='r')
ax.bar(x1, np.array(P_ANN_homer_FL), width, alpha=0.7, label='Homogeneity (RF)', color='deepskyblue')
ax.bar(x1 + width, np.array(P_ANN_heterogeneity), width, alpha=0.7, label='Heterogeneity', color='b')

plt.plot(x1, p_labels, color='y', linewidth=5.0, linestyle = '--', label='Sample Proportion(unbalanced)')
plt.plot(x1 + width, p_balanced, color='lightsalmon', linewidth=5.0, linestyle= '-', label='Sample Proportion (balanced)')
'''

'''
# ANN Se 
ax.bar(x1, np.array(Se_ANN_raw), width, alpha=0.7, label='Full-unbalanced (RF)', color='maroon')
ax.bar(x1 + width, np.array(Se_ANN_blanced), width, alpha=0.7, label='Full-balanced (RF)', color='r')
ax.bar(x1, np.array(Se_ANN_homer_FL), width, alpha=0.7, label='Homogeneity (RF)', color='deepskyblue')
ax.bar(x1 + width, np.array(Se_ANN_heterogeneity), width, alpha=0.7, label='Heterogeneity', color='b')

plt.plot(x1, p_labels, color='y', linewidth=5.0, linestyle = '--', label='Sample Proportion(unbalanced)')
plt.plot(x1 + width, p_balanced, color='lightsalmon', linewidth=5.0, linestyle= '-', label='Sample Proportion (balanced)')
'''

'''
# CNN Precision
ax.bar(x1, np.array(P_CNN_raw), width, alpha=0.7, label='Full-unbalanced (RF)', color='maroon')
ax.bar(x1 + width, np.array(P_CNN_blanced), width, alpha=0.7, label='Full-balanced (RF)', color='r')
ax.bar(x1, np.array(P_CNN_homer_FL), width, alpha=0.7, label='Homogeneity (RF)', color='deepskyblue')
ax.bar(x1 + width, np.array(P_CNN_heterogeneity), width, alpha=0.7, label='Heterogeneity', color='b')

plt.plot(x1, p_labels, color='y', linewidth=5.0, linestyle = '--', label='Sample Proportion(unbalanced)')
plt.plot(x1 + width, p_balanced, color='lightsalmon', linewidth=5.0, linestyle= '-', label='Sample Proportion (balanced)')
'''


# CNN Se 
ax.bar(x1, np.array(Se_CNN_raw), width, alpha=0.7, label='Full-unbalanced (FNN)', color='maroon')
ax.bar(x1 + width, np.array(Se_CNN_blanced), width, alpha=0.7, label='Full-balanced (FNN)', color='r')
ax.bar(x1, np.array(Se_CNN_homer_FL), width, alpha=0.7, label='Homogeneity (FNN)', color='deepskyblue')
ax.bar(x1 + width, np.array(Se_CNN_heterogeneity), width, alpha=0.7, label='Heterogeneity', color='b')

plt.plot(x1, p_labels, color='y', linewidth=5.0, linestyle = '--', label='Sample Proportion(unbalanced)')
plt.plot(x1 + width, p_balanced, color='lightsalmon', linewidth=5.0, linestyle= '-', label='Sample Proportion (balanced)')


# Customize the plot
ax.set_xticks(x1 + width / 2)  # Set the x-ticks at specified positions
class_labels = ["Normal", "VD", "CAD"]
ax.set_xticklabels(class_labels, fontsize=16, fontname='Times New Roman')

ytick_positions = np.arange(0.0, 1.1, 0.2)  # Define desired y-tick positions
ax.set_yticks(ytick_positions)
ax.set_yticklabels([f'{tick:.1f}' for tick in ytick_positions], fontsize=48, fontname='Times New Roman')

plt.scatter(x1, p_labels, color='deepskyblue', s=200)
plt.scatter(x1 + width, p_balanced, color='k', s=200)

plt.xlabel('Class', fontsize=16, fontname='Times New Roman')
plt.ylabel('Se (Recall) [%]', fontsize=16, fontname='Times New Roman')
# plt.legend(bbox_to_anchor=(1.05, 0.5), loc=3, borderaxespad=0)

# plt.legend(bbox_to_anchor=(1.05, 0.5), loc=3, borderaxespad=0)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderaxespad=0., fontsize=26)

font = FontProperties(family='Times New Roman', size=36)
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), borderaxespad=0., prop=font,  frameon=False)


plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()

#plt.subplots_adjust(bottom=0.4)
plt.savefig('Se0_FNN3.eps')
plt.show()
