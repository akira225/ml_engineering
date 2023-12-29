import warnings
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')


def train_best_model(X_train, X_val, y_train, y_val):
    C = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1, 2, 5, 10]
    models = []
    scores = []
    for reg_coef in C:
        model = LogisticRegression(C=reg_coef)
        models.append(model)
        model.fit(X_train, y_train)
        scores.append(model.score(X_val, y_val))
    best_i = np.argmax(scores)
    return models[best_i]


def save_model(model, X_val, y_val):
    with open('path.bin', 'wb') as f:
        pickle.dump(model, f)
    with open('path.txt', 'w') as f:
        f.write(classification_report(y_val, model.predict(X_val)))


if __name__ == '__main__':
    # argparse
    train = pd.read_csv('../data/preprocessed/train.csv')
    val = pd.read_csv('../data/preprocessed/val.csv')
    y_train = train['Survived']
    y_val = val['Survived']
    X_train = train.drop(['PassengerId', 'Survived'], axis=1, inplace=False).values
    X_val = val.drop(['PassengerId', 'Survived'], axis=1, inplace=False).values
    model = train_best_model(X_train, X_val, y_train, y_val)
    save_model(model, X_val, y_val)



