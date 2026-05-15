# feature_selection.py
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np

def select_features(X_train, y_train, X_val, X_test, k=750):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_k = selector.fit_transform(X_train, y_train)
    X_val_k = selector.transform(X_val)
    X_test_k = selector.transform(X_test)
    return X_train_k, X_val_k, X_test_k, selector

def plot_feature_scores(selector):
    scores = selector.scores_
    sorted_scores = np.sort(scores)[::-1]
    plt.figure(figsize=(10, 4))
    plt.plot(sorted_scores)
    plt.title("Feature Importance (F-score)")
    plt.xlabel("Feature Index (sorted by importance)")
    plt.ylabel("F-score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()