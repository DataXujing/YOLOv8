from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/last.pt")  # load a pretrained YOLOv8n model
model.export(format="onnx",opset=13)  # export the model to ONNX forma