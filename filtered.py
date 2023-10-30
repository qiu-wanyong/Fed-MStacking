import os
import shutil
import pandas as pd


# image_dir = 'image_cwt_raw/training-e'
# labels_file = 'image_cwt_raw/training-a/e_label.csv'

image_dir = 'CWT_data_resampled/training-e1'
labels_file = 'CWT_data_resampled/training-e/e_labels.csv'

'''  
# 重命名，删除.wav
file_list = os.listdir(image_dir)
# Rename image files
for filename in file_list:
    if filename.endswith('.wav.png'):
        new_filename = filename.replace('.wav.png', '.png')
        src_path = os.path.join(image_dir, filename)
        dst_path = os.path.join(image_dir, new_filename)
        os.rename(src_path, dst_path)
        print(f'Renamed {filename} to {new_filename}')

print("Image files renamed successfully.")

'''

image_filenames = [os.path.splitext(filename)[0] for filename in os.listdir(image_dir) if filename.endswith('.png')]

# Read the CSV file
labels_data = pd.read_csv(labels_file)

# Filter the rows based on image filenames
filtered_labels_data = labels_data[labels_data['filename'].isin(image_filenames)]

# Save the filtered DataFrame back to the CSV file
filtered_labels_data.to_csv(labels_file, index=False)

print("Labels filtered and saved successfully.")



image_filenames = sorted(os.listdir(image_dir))

# Rename and reindex the images
for idx, old_filename in enumerate(image_filenames, start=1):
    new_filename = f"e{idx:04d}.png"  # Format the new filename
    old_path = os.path.join(image_dir, old_filename)
    new_path = os.path.join(image_dir, new_filename)
    os.rename(old_path, new_path)  # Rename the file
print("Image filenames have been renamed and reindexed.")


labels_df = pd.read_csv(labels_file)

# Sort the DataFrame by the "filename" column
labels_df = labels_df.sort_values(by="filename")

# Update the "filename" column with new filenames
for idx, row in enumerate(labels_df.iterrows(), start=1):
    labels_df.loc[row[0], "filename"] = f"e{idx:04d}.png"  # Format the new filename

# Save the updated DataFrame back to the CSV file
labels_df.to_csv(labels_file, index=False)

print("Filenames in the label dataset have been renamed and reindexed.")










