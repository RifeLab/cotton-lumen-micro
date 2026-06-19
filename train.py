from ultralytics import YOLO
model = YOLO("yolo11m-seg.yaml")
yaml_file = "cotton-lumen-micro.yaml"
results = model.train(data=yaml_file, batch=-1, epochs=3000, imgsz=640)