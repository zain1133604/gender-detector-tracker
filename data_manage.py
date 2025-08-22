# 👨‍🔧👨‍🔧👨‍🔧 sub folder counting
# import os

# # # Define the path to your main folder
# folder_path = r'A:\\New folder\\human1'

# # Define image file extensions you want to count
# image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# count = 0
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         if file.lower().endswith(image_extensions):
#             count += 1

# print(f'Total images found: {count}')














# # 👨‍🔧👨‍🔧👨‍🔧👨‍🔧 organized 50k images like: getting all the images and labels from all directories and put them in one folder with separate folder for labels and iamge
# import os
# import shutil
# from PIL import Image

# source_dir = r"A:\\human_detection"
# target_dir = r"A:\\New folder\\project5_human_detection"
# images_target = os.path.join(target_dir, "images")
# labels_target = os.path.join(target_dir, "labels")

# os.makedirs(images_target, exist_ok=True)
# os.makedirs(labels_target, exist_ok=True)

# # Common image extensions
# image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"]

# # Counter for renaming
# counter = 1

# # Helper function to find label for an image
# def find_label(image_path):
#     # Look for labels folder in the same parent directories
#     parent = os.path.dirname(image_path)
#     while parent != source_dir and parent != os.path.dirname(parent):
#         for folder in os.listdir(parent):
#             folder_path = os.path.join(parent, folder)
#             if os.path.isdir(folder_path) and "label" in folder.lower():
#                 # Check if label exists
#                 label_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
#                 label_path = os.path.join(folder_path, label_name)
#                 if os.path.exists(label_path):
#                     return label_path
#         parent = os.path.dirname(parent)
#     return None

# # Walk through all files
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         file_lower = file.lower()
#         file_path = os.path.join(root, file)

#         if any(file_lower.endswith(ext) for ext in image_exts):
#             # Open and convert image to JPG
#             try:
#                 img = Image.open(file_path).convert("RGB")
#             except:
#                 print(f"Skipping corrupted file: {file_path}")
#                 continue

#             new_image_name = f"{counter}.jpg"
#             new_image_path = os.path.join(images_target, new_image_name)
#             img.save(new_image_path, "JPEG")

#             # Find corresponding label
#             label_path = find_label(file_path)
#             if label_path:
#                 new_label_name = f"{counter}.txt"
#                 new_label_path = os.path.join(labels_target, new_label_name)
#                 shutil.copyfile(label_path, new_label_path)
#             else:
#                 print(f"Label not found for {file_path}")

#             counter += 1

# print("All images and labels copied, renamed, and converted successfully!")








# import os
# from ultralytics import YOLO
# from PIL import Image
# import shutil

# # Paths
# source_dir = r"A:\\New folder\\human1"
# images_target = r"A:\\New folder\\project5_human_detection\\images"
# labels_target = r"A:\\New folder\\project5_human_detection\\labels"

# # YOLO model (pretrained person detector)
# model = YOLO("yolov8n.pt")  # You can also use yolov8s.pt or yolov8m.pt for better accuracy

# # Supported image extensions
# image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"]

# # Start counter from 50513
# counter = 50513

# # Function to convert image to jpg
# def convert_to_jpg(src_path, dst_path):
#     img = Image.open(src_path).convert("RGB")
#     img.save(dst_path, "JPEG")

# # Walk through all images in the source folder
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         if any(file.lower().endswith(ext) for ext in image_exts):
#             img_path = os.path.join(root, file)
            
#             # New image name
#             new_image_name = f"{counter}.jpg"
#             new_image_path = os.path.join(images_target, new_image_name)
            
#             # Convert and save image as jpg
#             convert_to_jpg(img_path, new_image_path)
            
#             # Run YOLOv8 detection on image
#             results = model.predict(new_image_path, imgsz=640, conf=0.25)  # conf=0.25 can be adjusted
            
#             # Save labels in YOLO format
#             label_file = os.path.join(labels_target, f"{counter}.txt")
#             with open(label_file, "w") as f:
#                 for result in results:
#                     boxes = result.boxes.xywhn.cpu().numpy()  # normalized x_center, y_center, w, h
#                     cls = result.boxes.cls.cpu().numpy()
#                     for c, box in zip(cls, boxes):
#                         # Only keep class 0 (person) if you want
#                         if int(c) == 0:
#                             x, y, w, h = box
#                             f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
#             counter += 1

# print("All images processed and labels generated successfully!")














# 😭👨‍🔧👨‍🔧👨‍🔧 making train test split
import os
import random
import shutil

# Base directories
base_dir = r"A:\\New folder\\project5_human_detection"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# New subfolders
splits = ["train", "validation", "test"]
for split in splits:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# Get all image files
all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(all_images)

def move_files(file_list, target_split):
    for img_file in file_list:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        # Destination paths
        img_dest = os.path.join(images_dir, target_split, img_file)
        label_dest = os.path.join(labels_dir, target_split, label_file)

        # Move only if label exists
        if os.path.exists(label_path):
            shutil.move(img_path, img_dest)
            shutil.move(label_path, label_dest)

# Step 1: Move 7k to validation
val_files = all_images[:7000]
move_files(val_files, "validation")

# Step 2: Move next 7k to test
test_files = all_images[7000:14000]
move_files(test_files, "test")

# Step 3: Move remaining to train
train_files = all_images[14000:]
move_files(train_files, "train")

print("✅ Dataset split completed successfully!")
