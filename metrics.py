import numpy

def confusion_matrix(actual, predicted):
    cm = numpy.zeros((3,3))
    for a, p in zip(actual, predicted):
        cm[a][p] += 1

    return cm