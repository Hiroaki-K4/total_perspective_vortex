import numpy as np


class CSP_ORI:
    def __init__(self, n_components=4):
        print("CSP class Init")
        self.n_components = n_components

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        print("n_classes: ", n_classes)
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2.")

        print(x)
