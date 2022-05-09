import shutil
import os
import random
import pickle

raw_folder = "E:\\LungData\\VINAI_Chest_Xray\\train\\train"

with open('file_list.pkl', 'rb') as f:
    file_list = pickle.load(f)


total_files = len(file_list)
print("Total files  = ", total_files)


train_folder = "E:\\LungData\\yolo_data\\images\\train"
val_folder = "E:\\LungData\\yolo_data\\images\\val"

train_labels_folder = "E:\\LungData\\yolo_data\\labels\\train"
val_labels_folder = "E:\\LungData\\yolo_data\\labels\\val"


total_files_validation = int(0.2 * total_files)  # 20% for validation
validaiton_files = random.choices(file_list, k=total_files_validation)
print("Total validation files = ", len(validaiton_files))

# Copy images và labels to validation folder
for file in validaiton_files:
    print("Validation file ", file)
    # Copy images
    shutil.copy(os.path.join(raw_folder, file), os.path.join(val_folder, file))

    # Copy labels
    shutil.copy(os.path.join(
        raw_folder, file[:-3] + 'txt'), os.path.join(val_labels_folder, file[:-3] + 'txt'))


# Copy images và labels to train folder
for file in file_list:
    if not (file in validaiton_files):
        print("Train file ", file)
        # Copy images
        shutil.copy(os.path.join(raw_folder, file),
                    os.path.join(train_folder, file))

        # Copy labels
        shutil.copy(os.path.join(
            raw_folder, file[:-3] + 'txt'), os.path.join(train_labels_folder, file[:-3] + 'txt'))
