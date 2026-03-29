import torch
from ultralytics import YOLO
import os

print("[INFO] Loading YOLOv8n model...")
yolo_model = YOLO("yolov8n.pt")

print("[INFO] Converting YOLO model to FP16...")
yolo_model.model.half()
torch.save(yolo_model.model.state_dict(), "yolov8n_fp16.pt")

print("[SUCCESS] YOLOv8n FP16 model saved as yolov8n_fp16.pt")
print("Original YOLOv8n.pt size:", os.path.getsize("yolov8n.pt") / 1e6, "MB")
print("Quantized FP16 size:", os.path.getsize("yolov8n_fp16.pt") / 1e6, "MB")

from midas.model_loader import load_midas
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform, net_w, net_h = load_midas(device)

print("[INFO] Converting MiDaS to FP16...")
model.half()
torch.save(model.state_dict(), "midas_fp16.pt")

print("[SUCCESS] MiDaS FP16 model saved as midas_fp16.pt")
