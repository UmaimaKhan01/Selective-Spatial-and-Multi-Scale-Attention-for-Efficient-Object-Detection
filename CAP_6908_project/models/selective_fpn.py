import sys
sys.path.insert(0, '/groups/mli2/CAP_6908_project/CAP_6908_project')

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class SpatialGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = max(1, in_channels // 4)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W] -> gate: [B, 1, H, W]
        return self.conv(x)


class ScaleGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = max(1, in_channels // 4)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W] -> scalar gate: [B, 1, 1, 1]
        gap = x.mean(dim=[2, 3])        # [B, C]
        s = self.fc(gap)                # [B, 1]
        return s.view(x.shape[0], 1, 1, 1)


class SelectiveFPNFasterRCNN(FasterRCNN):
    def __init__(self, num_classes, mode="full"):
        """
        mode: "spatial_only" | "scale_only" | "full"
        """
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights="DEFAULT"
        )
        super().__init__(backbone, num_classes=num_classes)

        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.mode = mode
        fpn_out_channels = self.backbone.out_channels  # 256 for ResNet50-FPN

        self.gate_levels = ["0", "1", "2", "3"]

        self.spatial_gates = nn.ModuleDict()
        self.scale_gates = nn.ModuleDict()

        for level in self.gate_levels:
            if mode in ("spatial_only", "full"):
                self.spatial_gates[level] = SpatialGate(fpn_out_channels)
            if mode in ("scale_only", "full"):
                self.scale_gates[level] = ScaleGate(fpn_out_channels)

    def _apply_gates(self, features):
        """
        Apply spatial and/or scale gates to FPN feature maps.
        Returns gated features dict and gate_stats dict (mean gate per level).
        """
        gated = {}
        gate_stats = {}

        for level, feat in features.items():
            if level not in self.gate_levels:
                gated[level] = feat
                continue

            gate = torch.ones(
                feat.shape[0], 1, feat.shape[2], feat.shape[3],
                device=feat.device, dtype=feat.dtype
            )

            if level in self.spatial_gates:
                spatial_g = self.spatial_gates[level](feat)
                gate = gate * spatial_g

            if level in self.scale_gates:
                scale_g = self.scale_gates[level](feat)
                gate = gate * scale_g

            gated[level] = feat * gate
            gate_stats[level] = gate.mean()

        return gated, gate_stats

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("Targets required during training.")

        original_image_sizes = []
        for img in images:
            original_image_sizes.append((img.shape[-2], img.shape[-1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        features, gate_stats = self._apply_gates(features)

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets
        )
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )

        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses, gate_stats

        return detections, gate_stats


def get_selective_model(num_classes, mode="full"):
    return SelectiveFPNFasterRCNN(num_classes=num_classes, mode=mode)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for mode in ["spatial_only", "scale_only", "full"]:
        model = get_selective_model(num_classes=91, mode=mode).to(device)
        model.eval()
        dummy = [torch.rand(3, 480, 640).to(device)]
        with torch.no_grad():
            dets, stats = model(dummy)
        print(f"[{mode}] Forward pass OK | detections: {len(dets[0]['boxes'])} | gate levels: {list(stats.keys())}")

    print("All selective model variants passed.")
