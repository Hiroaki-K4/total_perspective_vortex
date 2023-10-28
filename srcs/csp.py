import numpy as np
import scipy


class CSP_ORI:
    def __init__(self, n_components=4):
        print("CSP class Init")
        self.n_components = n_components

    def fit(self, x, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        if n_classes != 2:
            raise ValueError("n_classes must be 2.")

        covs = []
        for this_class in self.classes:
            x_class = x[y == this_class]
            # Convert data dimention to (channels, epoch * data_point)
            _, n_channels, _ = x_class.shape
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(n_channels, -1)
            print(x_class)
            print(x_class.shape)
            cov = np.dot(x_class, x_class.T)
            covs.append(cov)
            print(cov)
            print(cov.shape)

        # Solve generalized eigenvalue problem
        eig_val, eig_vec = scipy.linalg.eigh(covs[0], covs[1])
        for i in range(len(eig_vec)):
            eig_vec [i] = eig_vec[i] / np.linalg.norm(eig_vec[i])

        print("eig_val")
        print(eig_val)
        print(eig_val.shape)
        print("eig_vec")
        print(eig_vec)
        print(eig_vec.shape)
        i = np.argsort(eig_val)
        print(i)
        ix = np.empty_like(i)
        ix[1::2] = i[: len(i) // 2]
        ix[0::2] = i[len(i) // 2 :][::-1]
        print(ix)