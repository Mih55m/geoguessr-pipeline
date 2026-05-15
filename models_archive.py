def train_mlp_classifier(X_train, y_train, model_path="mlp_model.pkl"):
    print("Training MLP Classifier...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(300, 100),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        early_stopping=True,
        random_state=42,
        max_iter=200
    )
    mlp.fit(X_train, y_train)
    joblib.dump(mlp, model_path)
    print(f"MLP model saved to {model_path}")
    return mlp

# Bagging Classifier with SVM
def train_bagging_svm(base_model, X_train, y_train, model_path="bagging_svm_model.pkl"):
    print("Training BaggingClassifier with SVM base...")
    bagging = BaggingClassifier(
        estimator=base_model,
        n_estimators=5,
        max_samples=0.8,
        random_state=42,
        n_jobs=-1
    )
    bagging.fit(X_train, y_train)
    joblib.dump(bagging, model_path)
    print(f"Bagging SVM saved to {model_path}")
    return bagging


# VotingClassifier
def train_voting_classifier(models_dict, X_train, y_train, model_path="ensemble_model.pkl"):
    print("Training VotingClassifier with:", list(models_dict.keys()))
    estimators = [(name, model) for name, model in models_dict.items()]
    ensemble = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, model_path)
    print(f"Ensemble model saved to {model_path}")
    return ensemble