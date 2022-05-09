import pandas as pd
import cv2
import os
import pickle

# read csv file
csv_file = "E:\\LungData\\VINAI_Chest_Xray\\train_downsampled.csv"
df = pd.read_csv(csv_file)

print(df.head())

# files iteration in train directory
raw_folder = "E:\\LungData\\VINAI_Chest_Xray\\train\\train"
idx = 0
file_list = []
for file in os.listdir(raw_folder):
    if file[0] != ".":  # Ignore temp file
        print("Solving file {}- {}".format(idx, file))
        idx += 1
        # Find labels and filter class id != 14
        df_find = df[(df.image_id == file[:-4]) & (df.class_id != 14)]

        if len(df_find) > 0:
            # Read file to calculate size
            raw_image = cv2.imread(os.path.join(raw_folder, file), 0)
            image_width, image_height = raw_image.shape[1], raw_image.shape[0]

            labels = []

            for index, row in df_find.iterrows():

                # Find center and size of bounding box
                box_width = row[6] - row[4]
                box_height = row[7] - row[5]
                box_center_x = (row[6] + row[4]) / 2
                box_center_y = (row[7] + row[5]) / 2

                # normalization
                box_width_normalized = box_width / image_width
                box_height_normalized = box_height / image_height
                box_center_x_normalized = box_center_x / image_width
                box_center_y_normalized = box_center_y / image_height

                # Write labels to list
                labels.append([row[2], box_center_x_normalized, box_center_y_normalized,
                              box_width_normalized, box_height_normalized])

            # Iterate list to write labels to file
            txt_file = file[:-4] + ".txt"
            with open(os.path.join(raw_folder, txt_file), 'w') as f:
                for label in labels:
                    f.write('{} {} {} {} {}\n'.format(
                        label[0], label[1], label[2], label[3], label[4]))
            print("Done label ", txt_file)
            file_list.append(file)
            print("Number of files have label = ", len(file_list))

# write file_list to file pickle
with open('file_list.pkl', 'wb') as f:
    pickle.dump(file_list, f)
