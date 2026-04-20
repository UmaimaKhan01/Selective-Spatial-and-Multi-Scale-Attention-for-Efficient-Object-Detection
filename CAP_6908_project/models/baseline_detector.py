import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_baseline_model(num_classes):
    """
    Returns a Faster R-CNN ResNet50-FPN model with pretrained weights.
    num_classes: total number of classes INCLUDING background (index 0).
    For COCO, pass num_classes=91.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = get_baseline_model(num_classes=91)
    model.to(device)
    model.eval()

    dummy_images = [torch.rand(3, 480, 640).to(device)]
    with torch.no_grad():
        output = model(dummy_images)

    print("Forward pass OK.")
    print("Output keys:", list(output[0].keys()))
    print("Num detections:", len(output[0]["boxes"]))
