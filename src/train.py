import warnings
import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--report_dir", required=True)
    args = parser.parse_args()
    return args


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


def save_model(model_path, report_path, model, X_val, y_val):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(report_path, 'w') as f:
        f.write(classification_report(y_val, model.predict(X_val)))


if __name__ == '__main__':
    args = parse_arguments()
    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    y_train = train['Survived']
    y_val = val['Survived']
    X_train = train.drop(['PassengerId', 'Survived'], axis=1, inplace=False).values
    X_val = val.drop(['PassengerId', 'Survived'], axis=1, inplace=False).values
    model = train_best_model(X_train, X_val, y_train, y_val)
    save_model(args.model_dir, args.report_dir, model, X_val, y_val)
