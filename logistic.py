import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            row = np.insert(row, 0, np.ones(row.shape[0]), axis=1)
        t = row.dot(coef_)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        n = X_train.shape[0]
        if self.fit_intercept:
            X_train = np.insert(X_train, 0, np.ones(n), axis=1)
        self.coef_ = np.zeros(X_train.shape[1])
        self.fit_intercept = False

        for _ in range(self.n_epoch):
            yhat = self.predict_proba(X_train, self.coef_)
            dJdt = (yhat - y_train) * yhat * (1 - yhat)
            self.coef_ += -self.l_rate * dJdt.dot(X_train)
        self.fit_intercept = True

    def fit_log_loss(self, X_train, y_train):
        n = X_train.shape[0]
        if self.fit_intercept:
            X_train = np.insert(X_train, 0, np.ones(n), axis=1)
        self.coef_ = np.zeros(X_train.shape[1])
        self.fit_intercept = False

        for _ in range(self.n_epoch):
            yhat = self.predict_proba(X_train, self.coef_)
            y_error = (yhat - y_train)
            self.coef_ += -self.l_rate * y_error.dot(X_train) / n
        self.fit_intercept = True


    def predict(self, X_test, cut_off=0.5):
        yhat = self.predict_proba(X_test, self.coef_)
        return (yhat >= cut_off).astype(int)


def load_data():
    data = datasets.load_breast_cancer()
    col_names = ['worst concave points', 'worst perimeter', 'worst radius']#
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


model = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
model.fit_log_loss(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_pred, y_test)
res = {'coef_': list(model.coef_), 'accuracy': acc}
print(res)
