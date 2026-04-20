import os
import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import torch
from torch.utils.data import Dataset
from PIL import Image
import json


class LVISDetection(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        """
        LVIS v1 detection dataset.
        img_folder: path to COCO images (train2017 or val2017) — LVIS reuses COCO images
        ann_file:   path to lvis_v1_train.json or lvis_v1_val.json
        """
        self.img_folder = img_folder
        self.transforms = transforms

        with open(ann_file, 'r') as f:
            lvis = json.load(f)

        # Build image id -> image info map
        self.images = {img['id']: img for img in lvis['images']}

        # Build image id -> list of annotations map
        self.annotations = {}
        for ann in lvis['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Only keep images that have at least one annotation
        self.ids = [img_id for img_id in self.images.keys()
                    if img_id in self.annotations]

        # Build category id -> contiguous index map (LVIS has 1203 categories, non-contiguous IDs)
        self.cat_ids = sorted([cat['id'] for cat in lvis['categories']])
        self.cat_id_to_idx = {cat_id: idx + 1 for idx, cat_id in enumerate(self.cat_ids)}
        # idx 0 is background

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]

        # LVIS stores the coco_url or file_name — extract just the filename
        file_name = img_info['coco_url'].split('/')[-1]
        img_path = os.path.join(self.img_folder, file_name)
        img = Image.open(img_path).convert("RGB")

        annos = self.annotations.get(img_id, [])

        boxes = []
        labels = []
        areas = []

        for ann in annos:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            # Map LVIS category id to contiguous index
            labels.append(self.cat_id_to_idx[ann['category_id']])
            areas.append(ann['area'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)

        # LVIS does not have iscrowd — set to zeros
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


if __name__ == "__main__":
    from torchvision import transforms

    PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'

    dataset = LVISDetection(
        img_folder=os.path.join(PROJECT_ROOT, 'data', 'coco', 'train2017'),
        ann_file=os.path.join(PROJECT_ROOT, 'data', 'lvis', 'lvis_v1_train.json'),
        transforms=transforms.ToTensor()
    )

    print("LVIS train dataset size:", len(dataset))
    img, target = dataset[0]
    print("Image shape:", img.shape)
    print("Boxes shape:", target['boxes'].shape)
    print("Labels shape:", target['labels'].shape)
    print("Num categories (including background):", max(dataset.cat_id_to_idx.values()) + 1)
    print("LVIS sanity check PASSED.")
