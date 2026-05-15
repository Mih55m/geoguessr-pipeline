import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import copy

def get_data_loaders(data_dir, batch_size=32):
    """
    Creates streaming DataLoaders for Training and Testing.
    """
    print(f"Initializing PyTorch Data Pipeline from: {data_dir}")

    # 1. The Transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(num_ops=2, magnitude=9), # State-of-the-art randomized distortion
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the Dataset (WITHOUT transforms initially)
    full_dataset = datasets.ImageFolder(root=data_dir) 
    class_names = full_dataset.classes
    print(f"Found {len(full_dataset)} total images across {len(class_names)} classes.")

    # 3. Calculate Train/Test Split sizes
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # 4. Perform the Split
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 5. Apply the correct transforms to each split
    # We use copy so the test set doesn't accidentally inherit the train transforms
    train_dataset.dataset = copy.copy(full_dataset)
    train_dataset.dataset.transform = train_transform
    
    test_dataset.dataset = copy.copy(full_dataset)
    test_dataset.dataset.transform = test_transform

    # 6. Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, class_names

if __name__ == "__main__":
    # Updated to point to the SSD!
    DATA_PATH = r"C:\Geoguessr\dataset" 
    
    train_loader, test_loader, classes = get_data_loaders(DATA_PATH)
    
    images, labels = next(iter(train_loader))
    
    print("\n--- Pipeline Verification ---")
    print(f"Batch Image Shape: {images.shape} (Batch Size, Channels, Height, Width)")
    print(f"Batch Label Shape: {labels.shape}")
    print("Classes found:", classes)