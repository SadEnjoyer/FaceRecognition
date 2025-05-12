import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, to_absolute_path
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from Arcface import ArcFace
from utils import split_dataset_by_image_per_class as split_dataset
from dataset import FaceDataset

def evaluate_on_lfw(model, cfg, device, transform):
    from utils import get_embedding, align_face
    from ultralytics import YOLO
    import dlib
    import cv2
    from metrics.tarfar import calculate_tar_at_fars

    print("Evaluating on LFW...")

    model.eval()
    yolo_model = YOLO(to_absolute_path(cfg.inference.yolo_model_path))
    predictor = dlib.shape_predictor(to_absolute_path(cfg.inference.shape_predictor_path))

    scores = []
    gt_labels = []

    with torch.no_grad():
        with open(to_absolute_path(cfg.inference.pair_file), 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="LFW inference"):
            parts = line.strip().split()

            img1_name, img2_name, label = parts
            label = int(label)

            path1 = os.path.join(to_absolute_path(cfg.inference.img_dir), img1_name)
            path2 = os.path.join(to_absolute_path(cfg.inference.img_dir), img2_name)

            if not os.path.exists(path1) or not os.path.exists(path2):
                continue

            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)

            if img1 is None or img2 is None:
                continue

            results1 = yolo_model(path1, verbose=False)
            results2 = yolo_model(path2, verbose=False)

            boxes1 = results1[0].boxes.xyxy.cpu().numpy() if results1 and results1[0].boxes.xyxy.shape[0] else []
            boxes2 = results2[0].boxes.xyxy.cpu().numpy() if results2 and results2[0].boxes.xyxy.shape[0] else []

            if len(boxes1) == 0 or len(boxes2) == 0:
                continue

            try:
                aligned1 = align_face(img1, boxes1[0], predictor)
                aligned2 = align_face(img2, boxes2[0], predictor)
            except Exception:
                continue

            emb1 = get_embedding(aligned1, model, device, transform)
            emb2 = get_embedding(aligned2, model, device, transform)

            sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            scores.append(sim)
            gt_labels.append(label)

    print("LFW Results:")
    calculate_tar_at_fars(scores, gt_labels)



@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples, _ = split_dataset(
        root_dir=to_absolute_path(cfg.dataset.root_dir),
        val_ratio=cfg.dataset.val_ratio,
        seed=cfg.dataset.seed
    )

    print(f"Train samples: {len(train_samples)}")

    all_labels = [label for _, label in train_samples]
    num_classes = len(set(all_labels))
    print(f"Detected {num_classes} unique classes.")

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    train_dataset = FaceDataset(train_samples, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=4
    )

    model = instantiate(cfg.backbone).to(device)
    embedding_size = cfg.backbone.embedding_size
    arcface = ArcFace(cin=embedding_size, cout=num_classes, s=32.0, m=0.5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(model.parameters()) + list(arcface.parameters()),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay
    )

    total_steps = cfg.train.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    for epoch in range(cfg.train.epochs):
        print(f"\n=== Epoch {epoch + 1}/{cfg.train.epochs} ===")

        model.train()
        arcface.train()
        train_loss = 0.0
        correct_train = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = model(images)
            logits = arcface(embeddings, labels)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == labels).sum().item()

            train_bar.set_postfix(loss=loss.item())

        epoch_train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct_train / len(train_loader.dataset)

        print(f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        evaluate_on_lfw(model, cfg, device, transform)

        save_dir = to_absolute_path(cfg.train.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'arcface_state_dict': arcface.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth"))

if __name__ == "__main__":
    main()
