import os
import random
import shutil
from ultralytics import YOLO

# Define directories
image_dir = 'D:/IFGI_master/CameraTrap_final2/images/'
label_dir = 'D:/IFGI_master/CameraTrap_final2/labels/'
output_dir = 'D:/IFGI_master/CameraTrap_final2/'
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

# Define the train/validation split ratio
split_ratio = 0.8

# Create output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

# Load class names
with open(os.path.join(output_dir, 'classes.txt')) as f:
    class_names = [line.strip() for line in f]


# Loop over images
for image_name in os.listdir(image_dir):
    # Load label file
    label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
    with open(label_path) as f:
        label_lines = f.readlines()
    
    # Extract class indices
    class_indices = [int(line.split()[0]) for line in label_lines]
    
    # Determine which set to assign the image to
    if random.random() < split_ratio:
        output_subdir = train_dir
    else:
        output_subdir = val_dir
    
    # Copy image file
    image_path = os.path.join(image_dir, image_name)
    output_path = os.path.join(output_subdir, 'images', image_name)
    shutil.copy(image_path, output_path)
    
    # Copy label file
    output_path = os.path.join(output_subdir, 'labels', os.path.splitext(image_name)[0] + '.txt')
    shutil.copy(label_path, output_path)
    


model = YOLO("yolov8n.pt")
model.train(data = "D:/IFGI_master/CameraTrap_final2/data_custom.yaml", batch=9, epochs=50)
