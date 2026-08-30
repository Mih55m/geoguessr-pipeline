# Geoguessr AI Predictor 

# Geo-Spatial Landscape Classifier (Geoguessr AI)

An end-to-end computer vision and deep learning project that classifies geographic regions from street and landscape imagery using classical computer vision baselines, deep transfer learning (EfficientNet-B3), and an interactive Streamlit web dashboard.

---

## Project Architecture
Raw Images
   │
   ├──► [Classical ML Pipeline] ──► HOG / LBP / Color Histograms ──► PCA (500) ──► HalvingGridSearch SVM
   │
   └──► [Deep Learning Pipeline] ──► RandAugment ──► EfficientNet-B3 Backbone ──► Dynamic Weighted Cross-Entropy Loss
                                                                                      │
                                                                                      ▼
                                                                       Streamlit Web Interface (Interactive Inference)

---

## Key Features & Engineering Highlights

* **Classical Feature Engineering:** Extracted texture (Local Binary Patterns), edge/shape (Histogram of Oriented Gradients), and color histogram descriptors in parallel using `joblib.Parallel` across CPU cores.
* **Deep Transfer Learning:** Replaced the classification head of a pre-trained `EfficientNet-B3` convolutional neural network and fine-tuned it on CUDA GPU hardware using PyTorch.
* **Dynamic Class Imbalance Compensation:** Computed inverse class frequencies across training splits dynamically and passed them directly into `nn.CrossEntropyLoss(weight=...)` to handle imbalanced regional data.
* **Data Augmentation & Scheduling:** Implemented `RandAugment` distortions to enhance generalization and used a `CosineAnnealingLR` scheduler for smooth gradient descent convergence.
* **Interactive Web Dashboard:** Built a Streamlit interface that accepts image uploads and displays predicted regions alongside real-time probability confidence charts.

---

## Tech Stack

* **Core Frameworks:** PyTorch, Torchvision, Scikit-Learn, Streamlit
* **Computer Vision & Image Processing:** OpenCV, Scikit-Image, PIL
* **Data Manipulation & Acceleration:** NumPy, Pandas, Joblib (Multi-threading), CUDA
* **Model Architecture & Tuning:** EfficientNet-B3, Support Vector Machines (SVM), Principal Component Analysis (PCA), HalvingGridSearchCV, Cosine Annealing Learning Rate Scheduler

---

## How to Run Locally
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the UI: `streamlit run DeepLearning/app.py`
