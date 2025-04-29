import os
import argparse
import torch
from ultralytics import YOLO
import dlib
from concurrent.futures import ThreadPoolExecutor
from utils import process_image

parser = argparse.ArgumentParser(description="Preprocess face images with YOLO and alignment")
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--aligned_dir', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--shape_predictor_path', type=str, required=True)
parser.add_argument('--workers', type=int, default=8, help='Number of threads')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Используем устройство для YOLO: {device}")

yolo_model = YOLO(args.model_path).to(device)
predictor = dlib.shape_predictor(args.shape_predictor_path)

os.makedirs(args.aligned_dir, exist_ok=True)

for class_id in sorted(os.listdir(args.data_dir), key=lambda x: int(x)):
    class_dir = os.path.join(args.data_dir, class_id)
    if not os.path.isdir(class_dir):
        continue

    aligned_class_dir = os.path.join(args.aligned_dir, class_id)
    os.makedirs(aligned_class_dir, exist_ok=True)

    img_paths = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_image, img_path, aligned_class_dir, yolo_model, predictor) for img_path in img_paths]

print("Preprocessing завершён!")
