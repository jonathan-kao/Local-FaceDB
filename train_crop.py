import os
import pandas as pd
from PIL import Image
import ultralytics
from ultralytics import YOLO

ultralytics.checks()
model = YOLO('yolo/yolov8n-face.pt')

input_folder = "data/train"
output_folder = "data/train_224"
train_csv = 'csv/train.csv'
category_csv = 'csv/category.csv'

label_frame = pd.read_csv(train_csv)
filename_to_label = pd.Series(label_frame.Category.values, index=label_frame['File Name']).to_dict()

category_df = pd.read_csv(category_csv)
num_classes = category_df['Category'].nunique()
label_to_index = {row[1]: row[0] for row in category_df.itertuples(index=False)}

# Get filenames in order
filenames0 = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
filenames = []
for i in range(len(filenames0)):
    filenames.append(f'{i}.jpg')


def process_images_in_directory(input_folder, output_folder):
    detected = 0
    undetected = 0
    cant_load = 0
    list_len = len(filenames)

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    print('Start cropping...')
    num_crops = []
    for file_idx, filename in enumerate(filenames):

        if file_idx % 200 == 0:
            print(f'Process: {file_idx}/{list_len}')

        img_path = os.path.join(input_folder, filename)

        try:
            img = Image.open(img_path)
            img = img.convert("RGB")

        except Exception as e:
            print(f"Warning: Could not load image {filename} due to {e}. Skipping.")
            cant_load += 1
            continue

        # Detect face
        results = model.predict(img, verbose=False)
        boxes = results[0].boxes

        if boxes.cls.nelement() != 0:
            num_crops.append(boxes.cls.nelement())
            best_box = None
            highest_confidence = 0

            # Extract face with best confidence
            for box_idx, box in enumerate(boxes):
                x, y, w, h = [int(coord) for coord in box.xywh.tolist()[0]]
                confidence = boxes.conf[box_idx]
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_box = [x, y, w, h]

            x, y, w, h = best_box
            max_edge = max(w, h)
            x = max(x - max_edge // 2, 0)
            y = max(y - max_edge // 2, 0)
            cropped_img = img.crop((x, y, x + max_edge, y + max_edge))
            resized_img = cropped_img.resize((224, 224), Image.Resampling.LANCZOS)
            label = str(label_to_index[filename_to_label[filename]])
            save_dir = os.path.join(output_folder, label)

            # Create the output folder if it does not exist
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{label}_{len(os.listdir(save_dir))}.jpg")
            resized_img.save(save_path, "JPEG")
            detected += 1

        else:
            undetected += 1

    print(f"All files: {list_len} / Detected: {detected} / Undetected: {undetected} / Can't Load: {cant_load}")


process_images_in_directory(input_folder, output_folder)
