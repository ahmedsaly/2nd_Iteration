import os
import random
from ultralytics import YOLO

# Path to the folder containing the images
image_folder = "D:/IFGI_master/CameraTrap_final2/test_resized"

# Get a list of all the image files in the folder
image_files = os.listdir(image_folder)

# Randomly select 30 image files
selected_files = random.sample(image_files, 30)

# Create the YOLO model and make predictions on the selected images
model = YOLO("D:/IFGI_master/CameraTrap_final2/yolov8n_custom.pt")
for file in selected_files:
    image_path = os.path.join(image_folder, file)
    model.predict(source=image_path, show=True,max_det=1, line_thickness=3, conf=0.6)






