# evaluate.py
from sklearn.metrics import accuracy_score, classification_report
from joblib import load

def evaluate_model(model, X_test, y_test, class_names=None):
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names if class_names else None))

    return y_pred