import os
import yaml
import shutil
import random
import glob
from roboflow import Roboflow
from IPython.display import clear_output

# Set root path
root_path = os.getcwd()

# Clone the YOLOv8 repository
os.system("git clone https://github.com/airockchip/ultralytics_yolov8 ultralytics")
os.chdir("ultralytics")
os.system("git checkout 5b7ddd8f821c8f6edb389aa30cfbc88bd903867b")

# Install the YOLOv8 package
os.system("pip install -e .")
clear_output()

import ultralytics
ultralytics.checks()

# Create Training/images and Training/weights directories
os.makedirs(f"{root_path}/Training/images", exist_ok=True)
os.makedirs(f"{root_path}/Training/weights", exist_ok=True)

# Install Roboflow
os.system("pip install roboflow -q")

# Download the dataset
rf = Roboflow(api_key="api_key")
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
os.chdir(f"{root_path}/ultralytics")

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

with open("custom_yolov8.yaml", 'w') as f:
    f.write(custom_yolov8_config)

# Training settings
model = "yolov8n"
image_size = 640

# Train the model
os.system(f"yolo batch=192 epochs=700 patience=50 device=0 cache=ram task=detect mode=train model=./custom_{model}.yaml data={dataset.location}/data.yaml imgsz={image_size} plots=False pretrained=False single_cls={num_classes == '1'}")

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
print(latest)

# Export the model to ONNX
os.chdir(f"{root_path}/ultralytics")
os.system(f"yolo mode=export format=rknn model={latest}")

# Path to the ONNX model
ex_path = '.'.join(latest.split('.')[:-1]) + '.onnx'
print(ex_path)

# Install RKNN Toolkit 2
os.system("wget https://github.com/rockchip-linux/rknn-toolkit2/raw/2c2d03def0c0908c86985b8190e973976ecec74c/rknn-toolkit2/packages/rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl")
os.system("pip install ./rknn_toolkit2-1.6.0+81f21f4d-cp310-cp310-linux_x86_64.whl")

# Clone the RKNN Model Zoo repository
os.chdir(root_path)
os.system("git clone https://github.com/airockchip/rknn_model_zoo/")
os.chdir("rknn_model_zoo")
os.system("git checkout eaa94d6f57ca553d493bf3bd7399a070452d2774")
os.chdir("examples/yolov8/python")

# Create the imgs.txt file
imgs_txt_content = """
imgs/1.jpg
imgs/2.jpg
imgs/3.jpg
imgs/4.jpg
imgs/5.jpg
imgs/6.jpg
imgs/7.jpg
imgs/8.jpg
imgs/9.jpg
imgs/10.jpg
imgs/11.jpg
imgs/12.jpg
imgs/13.jpg
imgs/14.jpg
imgs/15.jpg
imgs/16.jpg
imgs/17.jpg
imgs/18.jpg
imgs/19.jpg
imgs/20.jpg
"""

with open("imgs.txt", 'w') as f:
    f.write(imgs_txt_content)

# Copy and rename images
def copy_and_rename_images(source_folder, destination_folder, n):
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    image_files = glob.glob(os.path.join(source_folder, '*.jpg'))
    selected_images = random.sample(image_files, min(n, len(image_files)))
    for i, image_path in enumerate(selected_images, start=1):
        destination_path = os.path.join(destination_folder, f'{i}.jpg')
        shutil.copy(image_path, destination_path)
    print(f"{min(n, len(image_files))} random images copied from '{source_folder}' to '{destination_folder}' and renamed.")

copy_and_rename_images(f"{dataset.location}/test/images", "imgs", 20)

# Update convert.py to use imgs.txt
convert_py_path = f"{root_path}/rknn_model_zoo/examples/yolov8/python/convert.py"
with open(convert_py_path, 'r') as f:
    convert_py = f.read()

convert_py = convert_py.replace('../../../datasets/COCO/coco_subset_20.txt', 'imgs.txt')

with open(convert_py_path, 'w') as f:
    f.write(convert_py)

# Perform quantization
to_quantize = True
quant_code = "i8" if to_quantize else "fp"
output_model = f"{root_path}/Training/weights/{dataset.name}-{model}-{image_size}-{quant_code}.rknn"

# Export to RKNN
os.chdir(f"{root_path}/rknn_model_zoo/examples/yolov8/python")
os.system(f"python convert.py {ex_path} rk3588 {quant_code} {output_model}")
