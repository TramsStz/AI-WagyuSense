"""
AI-WagyuSense: Dual-Head EfficientNet-B0 Model Architecture
Wat Khemaphirataram School - KIDE 2026 Project

Architecture:
  - Input: (B, 3, 224, 224) - Isolated Longissimus dorsi muscle patch
  - Shared Backbone: EfficientNet-B0 (feature vector dimension = 1280)
  - Head A (Classification): TAS Beef Quality Grade 1-5 (5 classes)
  - Head B (Regression): Marbling %IMF (continuous percentage)
"""

import torch
import torch.nn as nn
import torchvision.models as models

class DualHeadEfficientNetB0(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone_epochs=0):
        super(DualHeadEfficientNetB0, self).__init__()
        
        # Load EfficientNet-B0 backbone
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            
        in_features = self.backbone.classifier[1].in_features # 1280
        # Discard original single classifier
        self.backbone.classifier = nn.Identity()
        
        # Head A: TAS Quality Grade Classifier (Classes 1 to 5, mapped to 0..4)
        self.grade_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )
        
        # Head B: %IMF Marbling Regressor
        self.imf_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Extract features through EfficientNet-B0
        # features shape: (B, 1280, 7, 7) -> avgpool -> (B, 1280, 1, 1) -> flatten -> (B, 1280)
        feats = self.backbone.features(x)
        pooled = self.backbone.avgpool(feats)
        embedding = torch.flatten(pooled, 1)
        
        # Dual-Head Outputs
        grade_logits = self.grade_head(embedding)          # (B, 5)
        imf_pred = self.imf_head(embedding).squeeze(-1)    # (B,)
        
        return grade_logits, imf_pred

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadEfficientNetB0(pretrained=False).to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    logits, imf = model(dummy_input)
    print("Test Forward:")
    print("  Input Shape :", dummy_input.shape)
    print("  Logits Shape:", logits.shape)
    print("  IMF Shape   :", imf.shape)
