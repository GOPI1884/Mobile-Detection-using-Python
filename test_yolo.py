from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # This downloads and loads the model
print("Model loaded successfully!")