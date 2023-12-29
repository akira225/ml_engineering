import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
warnings.filterwarnings('ignore')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--train", required=True)
    args = parser.parse_args()
    return args


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)


def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)


def fill_missing_values(df_all: pd.DataFrame):
    # df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    df_all['Fare'] = df_all['Fare'].fillna(med_fare)
    df_all['Embarked'] = df_all['Embarked'].fillna('S')


def transform_features(data: pd.DataFrame):
    # Разделим пассажиров на группы по возрасту
    data['Age_band'] = 0
    data.loc[data['Age'] < 14, 'Age_band'] = 0
    data.loc[(data['Age'] >= 14) & (data['Age'] <= 21), 'Age_band'] = 1
    data.loc[(data['Age'] > 21) & (data['Age'] <= 35), 'Age_band'] = 2
    data.loc[(data['Age'] > 35) & (data['Age'] <= 55), 'Age_band'] = 3
    data.loc[data['Age'] > 55, 'Age_band'] = 4
    # Представим Fare в виде категории
    data['Fare_cat'] = 0
    data.loc[data['Fare'] <= 7.91, 'Fare_cat'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_cat'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare_cat'] = 2
    data.loc[(data['Fare'] > 31) & (data['Fare'] <= 513), 'Fare_cat'] = 3
    # Преобразуем строковые данные в числовые
    data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
    # сдвинем к нулю
    data['Pclass'] -= 1
    # уберем ненужные колонки
    data.drop(['Age', 'Fare', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


if __name__ == '__main__':
    args = parse_arguments()
    data = pd.read_csv(args.data)
    data_test = pd.read_csv(args.test_data)
    df_all = concat_df(data, data_test)
    fill_missing_values(df_all)
    transform_features(df_all)
    df_all = pd.get_dummies(df_all, columns=['Embarked',
                                             'Parch',
                                             'Pclass',
                                             'Sex',
                                             'SibSp',
                                             'Age_band',
                                             'Fare_cat'])
    train, test = divide_df(df_all)
    train, val = train_test_split(train, test_size=0.25, random_state=42)
    train.to_csv(args.train, index=False)
    val.to_csv(args.val, index=False)
    test.to_csv(args.test, index=False)
