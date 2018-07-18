import numpy as np
import csv
import math
import scipy.stats as stats
import pandas as pd
from decimal import *
import matplotlib.pyplot as plt


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

def hashing_trick(train_dataset, test_dataset, unique_values=20, cols_to_remove=[]):
    new_header = create_column(train_dataset)
    hashing_trick_columns = []
    for col in xrange(1,len(train_dataset) - 1):
        if len(set(train_dataset[col])) < unique_values and col not in cols_to_remove:
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

def create_column(dataset):
    new_header = []
    for col in xrange(len(dataset)):
        new_header.append("A" + str(col))
    return new_header

def remove_correlation(corr_df, df, eps=.05, col_to_remove=[]):
    import pdb; pdb.set_trace()
    indexes_to_remove = []
    for i, row in corr_df.iterrows():
        for j, value in row.iteritems():
            if abs(value) + eps >= 1 and i != j:
                indexes_to_remove.append(j)
    import pdb; pdb.set_trace()
    return indexes_to_remove

def get_na_cols_to_remove(analise):
    cols_to_remove = []
    for col, col_analise in analise.iteritems():
        if col_analise['na_percent'] >= Decimal(.80):
            cols_to_remove.append(col)
    return cols_to_remove

header, train_dataset = read_dataset('dataset/train.csv', 81, remove_header=True, delimiter=',')
test_header, test_dataset = read_dataset('dataset/test.csv', 80, remove_header=True, delimiter=',')

analise = analising_missing_values(train_dataset)
cols_to_remove = get_na_cols_to_remove(analise)

hashing_trick(train_dataset, test_dataset, cols_to_remove=cols_to_remove)

train_df = pd.DataFrame.from_csv('dataset/processed/trainWithHashTrick.csv', index_col=0, header=None)

# remove NA columns
na_quantity_by_columns = train_df.isnull().sum()
cols_to_remove = []
for index, value in na_quantity_by_columns.iteritems():
    if Decimal(value)/ len(train_df) >= .80:
        cols_to_remove.append(index)
for i in cols_to_remove:
    del train_df[i]
import pdb; pdb.set_trace()
filled_train_df = train_df.fillna(train_df.mean())
corr_df = filled_train_df.corr(method='pearson')
plt.matshow(corr_df)
plt.show()
# indexes_to_remove = remove_correlation(corr_df, train_df, col_to_remove=cols_to_remove)

 
