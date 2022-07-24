import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.first_mse = []
        self.last_mse = []
        self.first_log_loss = []
        self.last_log_loss = []

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            row = np.insert(row, 0, np.ones(row.shape[0]), axis=1)
        t = row.dot(coef_)
        return self.sigmoid(t)

    def mse(self, X, y):
        yhat = self.predict_proba(X, self.coef_)
        return ((yhat - y) ** 2).mean()

    def update_coef_mse(self, row, yval):
        yhat = self.predict_proba(row, self.coef_)
        dJdt = (yhat - yval) * yhat * (1 - yhat)
        self.coef_ += -self.l_rate * dJdt * row


    def fit_mse(self, X_train, y_train):
        n = X_train.shape[0]
        if self.fit_intercept:
            X_train = np.insert(X_train, 0, np.ones(n), axis=1)
        self.coef_ = np.zeros(X_train.shape[1])
        self.fit_intercept = False

        for i in range(n):
            self.update_coef_mse(X_train[i], y_train[i])
            self.first_mse.append(self.mse(X_train, y_train))

        for epoch in range(self.n_epoch - 2):
            for i in range(n):
                self.update_coef_mse(X_train[i], y_train[i])

        for i in range(n):
            self.update_coef_mse(X_train[i], y_train[i])
            self.last_mse.append(self.mse(X_train, y_train))

        self.fit_intercept = True

    def log_loss(self, X, y):
        yhat = self.predict_proba(X, self.coef_)
        return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()

    def update_coef_log(self, row, yval, n):
        yhat = self.predict_proba(row, self.coef_)
        y_error = (yhat - yval)
        self.coef_ += -self.l_rate * y_error * row / n

    def fit_log_loss(self, X_train, y_train):
        n = X_train.shape[0]
        if self.fit_intercept:
            X_train = np.insert(X_train, 0, np.ones(n), axis=1)
        self.coef_ = np.zeros(X_train.shape[1])
        self.fit_intercept = False

        for i in range(n):
            self.update_coef_log(X_train[i], y_train[i], n)
            self.first_log_loss.append(self.log_loss(X_train, y_train))

        for epoch in range(self.n_epoch - 2):
            for i in range(n):
                self.update_coef_log(X_train[i], y_train[i], n)

        for i in range(n):
            self.update_coef_log(X_train[i], y_train[i], n)
            self.last_log_loss.append(self.log_loss(X_train, y_train))

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

model.fit_mse(X_train, y_train)
y_pred_mse = model.predict(X_test)

model.fit_log_loss(X_train, y_train)
y_pred_log = model.predict(X_test)

skmodel = LogisticRegression()
skmodel.fit(X_train, y_train)
y_pred_sk = model.predict(X_test)

print({'mse_accuracy': accuracy_score(y_pred_mse, y_test),
    'logloss_accuracy': accuracy_score(y_pred_log, y_test),
    'sklearn_accuracy': accuracy_score(y_pred_sk, y_test),
    'mse_error_first': model.first_mse,
    'mse_error_last': model.last_mse,
    'logloss_error_first': model.first_log_loss,
    'logloss_error_last': model.last_log_loss})

print("""Answers to the questions:
1) 0.0001
2) 0.0000
3) 0.00152
4) 0.0055
5) expanded
6) expanded
"""
)
# res = {'coef_': list(model.coef_), 'accuracy': acc}
# print(res)
