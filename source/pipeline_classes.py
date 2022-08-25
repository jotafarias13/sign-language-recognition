import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin

## Feature Selector
class FeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

## Processing Numerical Features
class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, normalize=True, image_height=80, image_width=80):
        self.normalize = normalize
        self.image_height = image_height
        self.image_width = image_width

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = []

        if not (X.shape[1]==self.image_height and X.shape[2]==self.image_width):
            for img in X:
                image = Image.fromarray(img)
                image = image.resize((self.image_height,self.image_width))
                img = np.array(image)
                X_copy.append(img)

            X_copy = np.array(X_copy)
        else:
            X_copy = X.copy()

        if self.normalize:
            X_copy = np.array(X_copy)
            X_copy = X_copy/255

        return X_copy
