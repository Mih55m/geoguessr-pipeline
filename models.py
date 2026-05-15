# models.py
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


def train_optimized_svm(X_train,y_train,model_path="best_svm_pipeline.pkl"):
    print("Building SVM Pipeline with PCA and Std Scaler...")
    pipeline=Pipeline([
        ('scaler',StandardScaler()),
        ('pca',PCA(n_components=500)),
        ('svm',SVC(class_weight='balanced'))
    ])

    param_grid={
        'svm__C':[0.1,1,10,100],
        'svm__kernel':['rbf'],
        'svm__gamma':['scale','auto']
    }
    
    print("Starting HalvingGridSearch")
    grid = HalvingGridSearchCV(
        pipeline, 
        param_grid=param_grid, 
        factor=2, 
        verbose=2, 
        n_jobs=-1 
    )

    grid.fit(X_train,y_train)
    print("Best params found:{grid.best_params_}")
    joblib.dump(grid.best_estimator_,model_path)

    return grid.best_estimator_


def load_model(model_path='best_svm_halvinggrid_model_kbest750.pkl'):
    return joblib.load(model_path)