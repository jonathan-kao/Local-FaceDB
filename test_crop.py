import os
import csv
from PIL import Image
import ultralytics
from ultralytics import YOLO

ultralytics.checks()
model = YOLO('yolo/yolov8n-face.pt')

input_folder = "data/test"
output_folder = "data/test_224"
crops_num_csv = "csv/crops_num.csv"

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
    cnt = 0
    num_crops = []
    for file_idx, filename in enumerate(filenames):
        img_list = []

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
            for box_idx, box in enumerate(boxes):
                x, y, w, h = [int(coord) for coord in box.xywh.tolist()[0]]
                max_edge = max(w, h)
                x = max(x - max_edge // 2, 0)
                y = max(y - max_edge // 2, 0)
                cropped_img = img.crop((x, y, x + max_edge, y + max_edge))
                resized_img = cropped_img.resize((224, 224), Image.Resampling.LANCZOS)
                save_path = os.path.join(output_folder, f"{cnt}.jpg")
                resized_img.save(save_path, "JPEG")
                cnt += 1
            detected += 1

        else:
            num_crops.append(1)
            resized_img = img.resize((224, 224), Image.Resampling.LANCZOS)
            img_list.append(resized_img)
            save_path = os.path.join(output_folder, f"{cnt}.jpg")
            resized_img.save(save_path, "JPEG")
            cnt += 1
            undetected += 1

    print(f"All files: {list_len} / Detected: {detected} / Undetected: {undetected} / Can't Load: {cant_load}")
    with open(crops_num_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'Crops'])
        row = 0
        for num in num_crops:
            writer.writerow([row, num])
            row += 1


process_images_in_directory(input_folder, output_folder)
