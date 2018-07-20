import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv
import math
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from decimal import *
from sklearn import linear_model

# DEFINED VARIABLES
test_size = .2
num_folds = 10

def read_dataset(dataset_name, number_columns, remove_header=False, delimiter=','):
    dataset = []
    header = []
    csv_file = open(dataset_name, 'rb')
    csv_reader = csv.reader(csv_file, delimiter=delimiter)
    if remove_header:
        row = csv_reader.next()
        if len(header) == 0:
            for column in row:
                header.append(column)
    for i in xrange(number_columns):
        dataset.append([])
    for row in csv_reader:
        col = 0
        for column in row:
            dataset[col].append(column)
            col += 1
    if remove_header:
        return (header, dataset)
    else:
        return dataset

def analising_missing_values(dataset):
    analise = {}
    for col in xrange(len(dataset)-1):
        possible_values = set(dataset[col])
        analise[col] = {'possible_values': len(possible_values), 'na_percent': Decimal(0)}
        if 'NA' in list(possible_values):
            analise[col]['na_percent'] = Decimal(dataset[col].count('NA')) / Decimal(len(dataset[col]))
    return analise

def create_column(dataset):
    new_header = []
    for col in xrange(len(dataset)):
        new_header.append("A" + str(col))
    return new_header

def hashing_trick(train_dataset, test_dataset, unique_values=20, cols_to_remove=[]):
    new_header = create_column(train_dataset)
    hashing_trick_columns = []
    for col in xrange(1,len(train_dataset) - 1):
        if len(set(train_dataset[col])) <= unique_values and col not in cols_to_remove:
            hashing_trick_columns.append(new_header[col])

    training_lines = len(train_dataset[0])
    test_lines = len(test_dataset[0])

    merged_dataset = np.concatenate((train_dataset[:-1], test_dataset), axis=1)

    data_frame = pd.DataFrame(data=np.transpose(merged_dataset), columns=new_header[:-1])

    for col in hashing_trick_columns:
        dummies = pd.get_dummies(data_frame[col])
        data_frame = pd.concat([data_frame, dummies], axis=1)
        data_frame.drop([col], inplace=True, axis=1)

    train_data_frame = pd.concat([data_frame[0:training_lines],
                                  pd.DataFrame(train_dataset[-1], columns=['salesPrice'])], axis=1)
    test_data_frame = data_frame[training_lines:]

    print(len(train_data_frame.columns))
    print(len(test_data_frame.columns))

    train_data_frame.to_csv('dataset/processed/trainWithHashTrick.csv', index = False, header=False)
    test_data_frame.to_csv('dataset/processed/testWithHashTrick.csv', index = False, header=False)

def remove_columns(columns_to_remove, train_df, test_df):
    for col in columns_to_remove:
        del train_df[col]
        del test_df[col]

def remove_single_value_columns(train_df, test_df):
    columns_to_remove = []
    for col in train_df:
        possible_values = set(train_df[col])
        if len(list(possible_values)) == 1 and col < len(train_df.columns):
            columns_to_remove.append(col)
    remove_columns(columns_to_remove, train_df, test_df)

def remove_correlation(train_df, test_df, eps=.05, save=False, corr_level=.8):
    corr_df = filled_train_df.corr(method='pearson')
    if save:
        plt.matshow(corr_df)
        plt.show()
    columns_to_remove = []
    for i, row in corr_df.iterrows():
        for col, value in row.iteritems():
            if abs(value) + eps >= corr_level and i != col and col < len(row)-1:
                columns_to_remove.append(col)
    remove_columns(list(set(columns_to_remove)), train_df, test_df)

def get_regression_models():
    models = [
        ('LR', linear_model.LinearRegression()),
        ('R', linear_model.Ridge()),
        ('Lo', linear_model.Lasso(alpha=.015)),
        ('La',linear_model.Lars(positive=True)),
        ('OMP', linear_model.OrthogonalMatchingPursuit()),
        ('BR', linear_model.BayesianRidge()),
    ]
    return models

def fit_models(features, target, num_folds, test_size=.2, save=False):
    results = {}
    a_train, a_test, b_train, b_test = train_test_split(features, target, test_size=test_size)
    for i, model in enumerate(get_regression_models()):
        m = model[1]
        kfold = KFold(n_splits=num_folds)
        result = cross_val_score(m, a_train, b_train, cv=kfold)
        m.fit(a_train, b_train)
        predicted = m.predict(filled_test_df)
        pred_df = pd.DataFrame(np.transpose(predicted), columns=['SalePrice'])\
                    .rename(index={x: x + 1461 for x in range(len(predicted))})
        pred_df.index.name = 'Id'
        if save:
            pred_df.to_csv('dataset/prediction/predict{}.csv'.format(model[0]), header=True)
        results[model[0]] = {
            'score': result.copy(),
            'predict': pred_df.copy()
        }
    return results


header, train_dataset = read_dataset('dataset/train.csv', 81, remove_header=True, delimiter=',')
test_header, test_dataset = read_dataset('dataset/test.csv', 80, remove_header=True, delimiter=',')

# analise = analising_missing_values(train_dataset)
# cols_to_remove = get_na_cols_to_remove(analise)

# hashing trick

hashing_trick(train_dataset, test_dataset, unique_values=25)#, cols_to_remove=cols_to_remove)

train_df = pd.DataFrame.from_csv('dataset/processed/trainWithHashTrick.csv', index_col=0, header=None)
test_df = pd.DataFrame.from_csv('dataset/processed/testWithHashTrick.csv', index_col=0, header=None)

filled_train_df = train_df.fillna(train_df.mean())
filled_test_df = test_df.fillna(test_df.mean())

# remove single value columns

remove_single_value_columns(filled_train_df, filled_test_df)

# correlation

remove_correlation(filled_train_df, filled_test_df)

# predicting
import pdb; pdb.set_trace()
features_all = filled_train_df.iloc[:,0:-1]
target = filled_train_df.iloc[:,-1]
predictions = fit_models(features_all, target, num_folds, save=True)

 
