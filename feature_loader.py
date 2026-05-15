# feature_loader.py
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

def load_features_and_labels(feature_path, label_path):
    X = np.load(feature_path)
    y = np.load(label_path)
    print("Loaded features and labels")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    return X, y

def print_distribution(name, y_subset, class_names):
    print(f"\nDistribution in {name}:")
    counts = Counter(y_subset)
    for class_idx, count in counts.items():
        print(f"{class_names[class_idx]}: {count} samples")

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    # Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Validation split from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test