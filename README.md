# Geoguessr AI Predictor 

## Overview
An end-to-end Deep Learning pipeline that classifies geographic locations from images. Built with PyTorch and deployed as a local web application using Streamlit. 

## The Architecture & Engineering
* **The Engine:** Transfer learning utilizing an `EfficientNet-B3` backbone.
* **Data Pipeline:** Handled a highly imbalanced dataset (10:1 ratio) by engineering **Dynamic Class Weights** into the CrossEntropyLoss function, ensuring minority regions were not ignored.
* **Optimization:** Implemented a **Cosine Annealing Learning Rate Scheduler** and **RandAugment** to push the model to a highly robust 76.5% accuracy across 9 global regions.

## How to Run Locally
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the UI: `streamlit run DeepLearning/app.py`