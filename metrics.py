import numpy

def confusion_matrix(actual, predicted):
    cm = numpy.zeros((3,3))
    for a, p in zip(actual, predicted):
        cm[a][p] += 1

    return cm

def accuracy(matrix):
    res = 0
    for i in range(len(matrix)):
        res += matrix[i][i]
    return res / matrix.sum()
