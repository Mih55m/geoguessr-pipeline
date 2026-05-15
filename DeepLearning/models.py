import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class GeoguessrModel(nn.Module):
    def __init__(self, num_classes=9):
        super(GeoguessrModel, self).__init__()
        
        print("Loading pre-trained backbone...")
        self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    model = GeoguessrModel(num_classes=9)
    
    dummy_tensor = torch.randn(32, 3, 224, 224)
    
    print("\nPushing dummy data through the network...")
    output = model(dummy_tensor)
    
    print("\n--- Architecture Verification ---")
    print(f"Input Shape: {dummy_tensor.shape}")
    print(f"Output Shape: {output.shape} (Expected: [32, 9])")
    print("If you see [32, 9], the model's head was successfully transplanted!")