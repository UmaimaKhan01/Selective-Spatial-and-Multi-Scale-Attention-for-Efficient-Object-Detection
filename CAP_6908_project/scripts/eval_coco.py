import os
import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json

from datasets.coco_dataset import COCODetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = '/groups/mli2/CAP_6908_project/CAP_6908_project'


def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate(model, device, root, ckpt_path, model_label="model"):
    val_dataset = COCODetection(
        img_folder=os.path.join(root, "val2017"),
        ann_file=os.path.join(root, "annotations", "instances_val2017.json"),
        transforms=transforms.ToTensor()
    )
    data_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"Evaluating {model_label}"):
            images = [img.to(device) for img in images]

            raw = model(images)
            # selective model returns (detections, gate_stats); baseline returns list
            if isinstance(raw, tuple):
                outputs, _ = raw
            else:
                outputs = raw

            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                # Convert [x1,y1,x2,y2] -> [x,y,w,h] for COCO format
                boxes_xywh = boxes.clone()
                boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

                for box, score, label in zip(boxes_xywh, scores, labels):
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box.tolist(),
                        "score": float(score)
                    })

    res_file = os.path.join(PROJECT_ROOT, f"coco_results_{model_label}.json")
    with open(res_file, "w") as f:
        json.dump(results, f)
    print(f"Results saved to {res_file}")

    ann_file = os.path.join(root, "annotations", "instances_val2017.json")
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(res_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    print(f"\n=== {model_label} ===")
    coco_eval.summarize()
    return coco_eval.stats[0]  # AP @ IoU=0.50:0.95


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "spatial_only", "scale_only", "full"])
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pth file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(PROJECT_ROOT, "data", "coco")

    if args.model == "baseline":
        from models.baseline_detector import get_baseline_model
        model = get_baseline_model(num_classes=91)
    else:
        from models.selective_fpn import get_selective_model
        model = get_selective_model(num_classes=91, mode=args.model)

    ap = evaluate(model, device, root, args.ckpt, model_label=args.model)
    print(f"\nFinal mAP (AP@0.50:0.95): {ap:.4f}")
