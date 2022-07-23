import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# write your code here
class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            t = coef_[0] + row.dot(coef_[1:])
        else:
            t = row.dot(coef_)
        return self.sigmoid(t)


def load_data():
    data = datasets.load_breast_cancer()
    col_names = ['worst concave points', 'worst perimeter']#
    col_inds = [np.where(data.feature_names == name)[0][0] for name in col_names]
    X = data.data[:,col_inds]
    y = data.target
    return (X, y)


def standardise_data(X):
    mean_vals = X.mean(axis=0)
    std_vals = X.std(axis=0)
    return (X - mean_vals) / std_vals


def split_data(X, y):
    return train_test_split(X, y, train_size=0.8, random_state=43)


X, y = load_data()
X = standardise_data(X)
X_train, X_test, y_train, y_test = split_data(X, y)

coeffs = np.array([0.77001597, -2.12842434, -2.39305793])

model = CustomLogisticRegression()

print(list(model.predict_proba(X_test[:10], coeffs)))

