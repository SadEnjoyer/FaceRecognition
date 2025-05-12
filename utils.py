import dlib
import random

from glob import glob
from collections import defaultdict
from PIL import Image
import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np

def align_face(image: np.ndarray, bbox: np.ndarray, predictor: dlib.shape_predictor, size: int = 150) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(int)
    rect = dlib.rectangle(x1, y1, x2, y2)
    landmarks = predictor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), rect)
    return dlib.get_face_chip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), landmarks, size=size)

def read_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    return image

def save_image(image: np.ndarray, path: str):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image_bgr)

def process_image(img_path: str, aligned_class_dir: str, yolo_model, predictor: dlib.shape_predictor, size: int = 150):
    import os
    from utils import read_image, align_face, save_image

    try:
        image = read_image(img_path)
    except ValueError as e:
        print(e)
        return

    results = yolo_model(img_path, verbose=False)
    aligned_face = None

    if not results or not results[0].boxes.xyxy.shape[0]:
        print(f"Лицо не найдено в файле: {img_path}")
        return

    boxes = results[0].boxes.xyxy.cpu().numpy()
    bbox = boxes[0]

    try:
        aligned_face = align_face(image, bbox, predictor, size=size)
    except Exception as e:
        print(f"Ошибка при выравнивании {img_path}: {e}")
        return

    if aligned_face is None:
        print(f"Лицо не найдено или ошибка выравнивания: {img_path}")
        return

    filename = os.path.basename(img_path)
    aligned_img_path = os.path.join(aligned_class_dir, filename)
    save_image(aligned_face, aligned_img_path)

def split_dataset_by_image_per_class(root_dir, val_ratio=0.2, seed=42):
    random.seed(seed)

    class_to_images = defaultdict(list)
    image_paths = glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)

    for path in image_paths:
        label = os.path.basename(os.path.dirname(path))
        class_to_images[label].append(path)

    train_samples = []
    val_samples = []

    for label, paths in class_to_images.items():
        random.shuffle(paths)
        val_count = int(len(paths) * val_ratio)
        val_images = paths[:val_count]
        train_images = paths[val_count:]

        val_samples.extend([(path, label) for path in val_images])
        train_samples.extend([(path, label) for path in train_images])

    return train_samples, val_samples







def get_embedding(image: np.ndarray, model, device, transform):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img).squeeze(0)
        return F.normalize(emb, p=2, dim=0)

