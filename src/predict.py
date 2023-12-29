import warnings
import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')


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
    # argparse
    test = pd.read_csv('../data/preprocessed/test.csv')
    ids = test['PassengerId']
    X_test = test.drop(['PassengerId'], axis=1, inplace=False).values
    model = load_model('path.bin')
    y = model.predict(X_test)
    save_predictions(path, ids, y)
