import os
import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.coco_dataset import COCODetection

PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'


def get_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    root = os.path.join(PROJECT_ROOT, "data", "coco")

    train_dataset = COCODetection(
        img_folder=os.path.join(root, "train2017"),
        ann_file=os.path.join(root, "annotations", "instances_train2017.json"),
        transforms=get_transform()
    )

    print("Train dataset size:", len(train_dataset))

    img, target = train_dataset[0]
    print("Image type:", type(img))
    print("Image shape:", img.shape)
    print("Target keys:", list(target.keys()))
    print("Boxes shape:", target["boxes"].shape)
    print("Labels shape:", target["labels"].shape)
    print("Image ID:", target["image_id"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    images, targets = next(iter(train_loader))
    print("\nBatch test:")
    print("Batch size:", len(images))
    print("First image shape:", images[0].shape)
    print("First target boxes:", targets[0]["boxes"].shape)
    print("\nSanity check PASSED.")
