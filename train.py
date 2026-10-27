from ultralytics import YOLO
model = YOLO("yolo11s-seg.pt")
yaml_file = "cotton-lumen-micro.yaml"
results = model.train(data=yaml_file, batch=-1, epochs=300, imgsz=960, overlap_masks=False, patience=30, cos_lr=True)
