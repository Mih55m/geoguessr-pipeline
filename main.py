#main.py
from data_loader import get_class_names, get_class_counts
from features import run_feature_extraction
from feature_loader import load_features_and_labels,print_distribution,split_data
from feature_selection import select_features, plot_feature_scores
from models import train_optimized_svm,load_model
from evaluate import evaluate_model
from visualization import plot_accuracy_vs_k,show_sample_predictions,plot_confusion_matrix,plot_accuracy_vs_c,plot_accuracy_by_kernel,plot_hyperparam_heatmap
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    #Loading the dataset
    class_names = get_class_names()
    class_counts = get_class_counts()

    print("Class Names:")
    for name in class_names:
        print("-", name)

    print("\nClass Counts:")
    for idx, count in class_counts.items():
        print(f"{class_names[idx]}: {count} images")


    # Running feature extraction
    # We do not want this running for every execution,
    # if we already have the features saved
    # run_feature_extraction(
    #     output_feature_path="G:/My Drive/GeoguessrClassifier/features.npy",
    #     output_label_path="G:/My Drive/GeoguessrClassifier/labels.npy"
    #     )


    # Load saved features
    X, y = load_features_and_labels(
        "./features.npy",
        "./labels.npy"
    )

    # Show first 5 examples
    for i in range(5):
        print(f"\nImage {i} — Label: {y[i]} ({class_names[y[i]]})")
        print("Feature vector (first 10 values):", X[i][:10])

    # Split into train, val and test
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,stratify=y,random_state=42)

    print_distribution("Training set", y_train,class_names)
    print_distribution("Test set", y_test,class_names)


    # Model training
    best_model=train_optimized_svm(X_train,y_train,model_path="best_svm_pipeline.pkl")

    # Evaluate the model
    y_pred = evaluate_model(best_model, X_test, y_test, class_names)

    # Showing results and metrics of the trained model
    show_sample_predictions(y_test, y_pred, class_names)
    plot_confusion_matrix(y_test, y_pred, class_names)

if __name__ == "__main__":
    main()