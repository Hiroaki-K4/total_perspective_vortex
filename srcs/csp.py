import numpy as

class CSP_ORI:
    def __init__(self, n_components=4):
        print("CSP class Init")

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        print("n_classes: ", n_classes)
