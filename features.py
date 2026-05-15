# feature_extraction.py
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm
from joblib import Parallel, delayed
from torchvision import datasets
import os

# Constants
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
DATA_DIR = "G:/My Drive/GeoguessrClassifier/compressed_dataset"

def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256]*3)
    return cv2.normalize(hist, hist).flatten()

def extract_hog_features(image, resize=(128, 128)):
    image = cv2.resize(image, resize)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

def extract_lbp_features(image, resize=(128, 128)):
    image = cv2.resize(image, resize)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_all_features(image_path, label):
    image = cv2.imread(image_path)
    if image is None:
        return None
    color_feat = extract_color_histogram(image)
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    full_feature = np.hstack([color_feat, hog_feat, lbp_feat])
    return full_feature, label

def run_feature_extraction(output_feature_path, output_label_path):
    dataset = datasets.ImageFolder(root=DATA_DIR)
    print("Extracting features...")

    results = Parallel(n_jobs=-1)(
        delayed(extract_all_features)(path, label) for path, label in tqdm(dataset.imgs)
    )

    # Filter out any None entries
    results = [r for r in results if r is not None]
    features, labels = zip(*results)
    X = np.array(features)
    y = np.array(labels)

    print(f"Extracted features from {len(X)} images. Feature vector shape: {X.shape}")

    np.save(output_feature_path, X)
    np.save(output_label_path, y)
    print("Features and labels saved.")