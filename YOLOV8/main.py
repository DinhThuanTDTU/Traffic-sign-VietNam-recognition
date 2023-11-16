from ultralytics import YOLO

# Load an official or custom model
model = YOLO('E:/code_project/pythonProject_yolov8/yolov8m/yolov8m_final.pt')  # Load an official Detect model


# Perform tracking with the model
results = model.track(source="C:/Users/boyph/Downloads/GX010319.mp4", show=True, tracker="bytetrack.yaml")