
import torch
import torch.nn as nn
import torchvision.models as models

class SolarMaintenanceModel(nn.Module):
    def __init__(self, model_name='resnet50', hidden_dim=256):
        super(SolarMaintenanceModel, self).__init__()
        
        # Using ResNet50 as a stable, high-performance backbone for industrial fault detection
        # This replaces the unstable FastSAM/MobileSAM integration
        print(f"Loading {model_name} backbone...")
        base_model = models.resnet50(pretrained=True)
        
        # Remove the final classification layer (fc)
        # ResNet50 output before fc is 2048 channels (Global Avg Pool already applied in standard forward? No, usually .avgpool then .fc)
        # We want the features. 
        # Ideally: distinct encoder. 
        # We will strip the last layer.
        self.backbone = nn.Sequential(*list(base_model.children())[:-1]) # Output: (B, 2048, 1, 1)
        self.backbone_dim = 2048
        
        # Regression Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.backbone(x) # (B, 2048, 1, 1)
        risk_score = self.head(features)
        return risk_score

    def freeze_backbone(self, freeze=True):
        print(f"Set Backbone Frozen: {freeze}")
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def freeze_backbone(self, freeze=True):
        """
        Phase 1: Freeze backbone.
        Phase 2: Unfreeze.
        """
        print(f"Set Backbone Frozen: {freeze}")
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
            
    def unfreeze_last_block(self):
        """
        Phase 2 Refinement: Unfreeze only the last layer/block of encoder.
        Implementation specific to ViT structure.
        """
        print("Unfreezing last block...")
        # Placeholder logic: unfreeze last named parameter group
        # Real implementation depends on ViT depth
        for param in self.backbone.parameters():
            param.requires_grad = True # Fully unfreeze for now or implement specific logic
