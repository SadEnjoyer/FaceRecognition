import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import cv2
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, to_absolute_path
import dlib
from ultralytics import YOLO
import wandb

from metrics.tarfar import calculate_tar_at_fars
from utils import get_embedding, align_face

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

@hydra.main(config_path="configs", config_name="config.yaml", version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.wandb.use:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    model = instantiate(cfg.backbone).to(device)

    checkpoint = torch.load(to_absolute_path(cfg.inference.model_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    yolo_model = YOLO(to_absolute_path(cfg.inference.yolo_model_path))

    predictor = dlib.shape_predictor(to_absolute_path(cfg.inference.shape_predictor_path))

    scores = []
    gt_labels = []

    with open(to_absolute_path(cfg.inference.pair_file), 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        parts = line.strip().split()
        if len(parts) != 3:
            continue

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
        except Exception as e:
            print(f"Ошибка при выравнивании лиц: {e}")
            continue

        emb1 = get_embedding(aligned1, model, device, transform)
        emb2 = get_embedding(aligned2, model, device, transform)

        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        scores.append(sim)
        gt_labels.append(label)

    print("Calculating TAR @ FAR")
    results = calculate_tar_at_fars(scores, gt_labels)

    if cfg.wandb.use:
        for far_label, stats in results.items():
            wandb.log({
                f"TAR@FAR={far_label}": stats["TAR"],
                f"Actual_FAR@{far_label}": stats["FAR"],
                f"Threshold@{far_label}": stats["threshold"]
            })

        wandb.log({
            "total_pairs": len(lines),
            "skipped_pairs": len(lines) - len(scores),
            "valid_pairs": len(scores)
        })

        wandb.finish()

if __name__ == "__main__":
    main()
