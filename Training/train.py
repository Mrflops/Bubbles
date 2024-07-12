import os
import yaml
from roboflow import Roboflow

# Set root path
root_path = os.getcwd()

# Function to create directories if they don't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Create necessary directories
create_dir(f"{root_path}/Training/images")
create_dir(f"{root_path}/Training/weights")

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="your_roboflow_api_key")  # Replace with your Roboflow API key
project = rf.workspace("trigon5990").project("conesandcubes")
dataset = project.version(6).download("yolov8")

# Update data.yaml
data_yaml_path = f"{dataset.location}/data.yaml"
with open(data_yaml_path, 'r') as f:
    data_yaml = f.read()

data_yaml = data_yaml.replace('test: ..', f'test: {dataset.location}')
data_yaml = data_yaml.replace('train: ', f'train: {root_path}/Training/images/')
data_yaml = data_yaml.replace('val: ', f'val: {root_path}/Training/images/')

with open(data_yaml_path, 'w') as f:
    f.write(data_yaml)

# Get number of classes
with open(data_yaml_path, 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])
print(f"num_classes: {num_classes}")

# Create custom YOLOv8 config
custom_yolov8_config = f"""
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: {num_classes}  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
activation: nn.ReLU()
# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
"""

with open(f"{root_path}/Training/custom_yolov8.yaml", 'w') as f:
    f.write(custom_yolov8_config)

# Training settings
model = "yolov8n"
image_size = 640

# Train the model
os.system(f"yolo batch=192 epochs=700 patience=50 device=0 cache=ram task=detect mode=train model=./Training/custom_{model}.yaml data={dataset.location}/data.yaml imgsz={image_size} plots=False pretrained=False single_cls={num_classes == '1'}")

# Find the latest model
latest_modified_time = 0
latest = None

for foldername, subfolders, filenames in os.walk(root_path):
    for filename in filenames:
        if filename == "best.pt":
            file_path = os.path.join(foldername, filename)
            modified_time = os.path.getmtime(file_path)
            if modified_time > latest_modified_time:
                latest_modified_time = modified_time
                latest = file_path
print(f"Best model saved at: {latest}")
