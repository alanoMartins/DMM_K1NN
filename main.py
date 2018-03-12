from sklearn import datasets
from sklearn.model_selection import train_test_split
from classifier import DMM, K1NN
from metrics import confusion_matrix, accuracy
from validation import CrossValidation



X, y = datasets.load_iris(return_X_y=True)


cr = CrossValidation(0.4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)


dmm = DMM()

dmm.train(X_train, y_train)
dmm_pred = dmm.predict(X_test)

dmm_cm = confusion_matrix(y_test, dmm_pred)

k1nn = K1NN()

k1nn.train(X_train, y_train)
k1nn_pred = list(k1nn.predict(X_test))

k1nn_cm = confusion_matrix(y_test, k1nn_pred)

print("Confusion Matrix:")
print("\n DMM: ")
print(dmm_cm)
print("Accuracy {}".format(accuracy(dmm_cm)))
print("\n K1NN: ")
print(k1nn_cm)
print("Accuracy {}".format(accuracy(k1nn_cm)))
