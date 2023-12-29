import warnings
import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    return args


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_predictions(path, ids, y):
    output = pd.DataFrame({
        'PassengerId': ids,
        'Survived': y
    })
    output.to_csv(path, index=False)


if __name__ == '__main__':
    args = parse_arguments()
    test = pd.read_csv(args.data)
    ids = test['PassengerId']
    X_test = test.drop(['PassengerId'], axis=1, inplace=False).values
    model = load_model(args.model)
    y = model.predict(X_test)
    save_predictions(args.out, ids, y)
