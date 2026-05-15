import torch
import torch.nn as nn
import torch.optim as optim
import time
import os

# Import our custom files!
from dataset import get_data_loaders
from models import GeoguessrModel

def train_model():
    # 1. Setup the Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Hardware Check ---")
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"----------------------\n")

    # 2. Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 15  # We will do 5 quick laps around the dataset
    LEARNING_RATE = 1e-4  # A gentle learning rate because the model is already pre-trained

    # Using a raw string (r"") so Windows backslashes are read correctly
    DATA_PATH = r"C:\Geoguessr\dataset"

    # 3. Initialize Data & Model
    train_loader, test_loader, classes = get_data_loaders(DATA_PATH, batch_size=BATCH_SIZE)
    
    model = GeoguessrModel(num_classes=len(classes))
    model = model.to(device) # BLAST THE MODEL TO THE GPU VRAM!

    # 4. The Loss Function and Optimizer
    # ==========================================
    # --- DYNAMIC WEIGHT CALCULATION ---
    # ==========================================
    print("Calculating dataset imbalance...")
    
    # 1. Get the original labels and the specific indices used for the Training Split
    all_targets = train_loader.dataset.dataset.targets
    train_indices = train_loader.dataset.indices
    
    # 2. Extract only the labels belonging to the training set
    train_targets = [all_targets[i] for i in train_indices]
    
    # 3. Count exactly how many images exist for each class dynamically
    class_counts = [train_targets.count(i) for i in range(len(classes))]
    total_train_samples = sum(class_counts)
    
    # 4. Calculate the mathematical weights (Total / Class Count)
    class_weights = [total_train_samples / c for c in class_counts]
    
    print(f"-> Class Counts: {class_counts}")
    print(f"-> Class Weights: {[round(w, 2) for w in class_weights]}")

    # 5. Apply the weights to the Loss Function
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    # ==========================================
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_accuracy = 0.0

    # 5. The Training Loop (The Engine)
    print("\n🔥 Starting Training Engine 🔥\n")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train() # Tell the model it is learning time (enables things like Dropout)
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            # Move the streaming data directly to the GPU
            images = images.to(device)
            labels = labels.to(device)

            # Step A: Clear old gradients from the last batch
            optimizer.zero_grad()
            
            # Step B: Forward pass (Make a guess!)
            outputs = model(images)
            
            # Step C: Calculate the error (How wrong was the guess?)
            loss = criterion(outputs, labels)
            
            # Step D: Backward pass (Calculates the calculus gradients to fix the error)
            loss.backward()
            
            # Step E: Update the weights (Learn!)
            optimizer.step()
            
            running_loss += loss.item()

            # Print an update every 100 batches
            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        scheduler.step()
        # --- EVALUATION PHASE (The Final Exam) ---
        model.eval() # Tell the model to freeze its weights for testing
        correct = 0
        total = 0
        
        # torch.no_grad() disables the massive calculus tracking, saving VRAM and speeding up testing
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                # Get the prediction (the index with the highest probability score)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_accuracy = 100 * correct / total
        epoch_time = time.time() - start_time
        
        print(f"\n=====================================")
        print(f"  Epoch {epoch+1} Summary  ")
        print(f"=====================================")
        print(f"Average Training Loss: {running_loss/len(train_loader):.4f}")
        print(f"Test Accuracy:         {epoch_accuracy:.2f}%")
        print(f"Time Taken:            {epoch_time:.2f} seconds")
        print(f"=====================================\n")

        # Save the model if it beat our previous high score
        if epoch_accuracy > best_accuracy:
            print(f"🏆 New Best Accuracy! Saving model to disk...\n")
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), "best_efficientB3_geoguessr.pth")

if __name__ == "__main__":
    train_model()