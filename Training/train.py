import os
import urllib.request
from rknn.api import RKNN
from ultralytics import YOLO


def train_yolov8_model(data_path, model_path, epochs=50, img_size=640):
    """Train a YOLOv8 model using the Ultralytics library."""
    # Initialize YOLO model
    model = YOLO('yolov8n.yaml')  # You can specify a different model configuration if needed

    # Train the model
    model.train(data=data_path, epochs=epochs, imgsz=img_size)

    # Save the trained model
    model.save(model_path)

def download_model(url, model_path):
    """Download the YOLOv8 model if it doesn't exist locally."""
    if not os.path.exists(model_path):
        print(f"Downloading YOLOv8 model from {url}")
        urllib.request.urlretrieve(url, model_path)
    else:
        print(f"Model already exists at {model_path}")


def convert_model_to_rknn(model_path, rknn_path):
    """Convert YOLOv8 model to RKNN format."""
    # Initialize RKNN object
    rknn = RKNN()

    # Load the ONNX model
    print("--> Loading model")
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print("Load model failed!")
        return
    print("done")

    # Preprocess config
    print("--> Building model")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        reorder_channel="0 1 2"
    )

    # Build the model
    ret = rknn.build(do_quantization=True)
    if ret != 0:
        print("Build model failed!")
        return
    print("done")

    # Export the RKNN model
    print("--> Export RKNN model")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print("Export RKNN model failed!")
        return
    print("done")


def main():
    # Define paths and parameters
    data_path = 'path/to/your/dataset.yaml'  # Path to your dataset.yaml file
    trained_model_path = 'yolov8_trained.pt'
    onnx_model_path = 'yolov8.onnx'
    rknn_model_path = 'yolov8.rknn'

    # Train YOLOv8 model
    train_yolov8_model(data_path, trained_model_path)

    # Convert trained model to ONNX format
    model = YOLO(trained_model_path)
    model.export(format='onnx')

    # Convert YOLOv8 ONNX model to RKNN
    convert_model_to_rknn(onnx_model_path, rknn_model_path)


if __name__ == "__main__":
    main()
