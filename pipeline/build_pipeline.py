
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def build_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    return pipeline