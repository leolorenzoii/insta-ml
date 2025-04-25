"""Module containing helper funcitons for model evaluation"""

import pandas as pd


class AutogluonWrapper:
    """Wrapper to allow the use of SHAP on AutoGluon models"""
    def __init__(self, predictor, feature_names):
        self.ag_model = predictor
        self.feature_names = feature_names
    
    def predict_binary_prob(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1, -1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)

        return self.ag_model.predict_proba(X, as_multiclass=False)
