import numpy as np
import csv
import math
import scipy.stats as stats
import pandas as pd


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

def remove_correlation(test_dataset, train_dataset, header, eps=.05):
    columns_to_remove = []
    indexes_to_remove = []
    headers_to_remove = []

    for i in xrange(len(test_dataset)):
        if len(set(train_dataset[i])) == 1:
            indexes_to_remove.append(i)
            columns_to_remove.append(train_dataset[i])
            headers_to_remove.append(header[i])
            continue
        if i not in indexes_to_remove:
            for j in xrange(i, len(test_dataset)):
                if i != j  and j not in indexes_to_remove:
                    correlation = stats.pearsonr(train_dataset[i], train_dataset[j])[0]
                    if abs(correlation) + eps >=  1:
                        indexes_to_remove.append(j)
                        columns_to_remove.append(train_dataset[j])
                        headers_to_remove.append(header[j])
    for col in columns_to_remove:
        index = train_dataset.index(col)
        train_dataset.remove(col)
        del test_dataset[index]

    for h in headers_to_remove:
        header.remove(h)

def hashing_trick(train_dataset, test_dataset, unique_values=20):
    new_header = create_column(train_dataset)
    hashing_trick_columns = []
    for col in xrange(len(train_dataset) - 1):
        if len(set(train_dataset[col])) < unique_values:
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
    print train_data_frame
    test_data_frame = data_frame[training_lines:]

    print(len(train_data_frame.columns))
    print(len(test_data_frame.columns))

    train_data_frame.to_csv('dataset/processed/trainWithHashTrick.csv', index = False, header= False)
    test_data_frame.to_csv('dataset/processed/testWithHashTrick.csv', index = False, header= False)

def create_column(dataset):
    new_header = []
    for col in xrange(len(dataset)):
        new_header.append("A" + str(col))
    return new_header

header, train_dataset = read_dataset('dataset/train.csv', 81, remove_header=True)
test_header, test_dataset = read_dataset('dataset/test.csv', 80, remove_header=True)


hashing_trick(train_dataset, test_dataset)

# header, train_dataset = read_dataset('dataset/train.csv', 582, remove_header=True)
# test_dataset = read_dataset('dataset/test.csv', 80, remove_header=False)
# remove_correlation(test_dataset, train_dataset, header=.01)



