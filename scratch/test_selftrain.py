import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC

LABELED_PROP = 0.2

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 5)

u = np.random.rand(y_train.shape[0])
y_sl = np.copy(y_train)
y_sl[u >= LABELED_PROP] = -1


print(f"Training with {sum(u < LABELED_PROP)} labeled observations")
base_classifier = SVC(kernel="rbf", gamma=0.5, probability=True)
s30 = SelfTrainingClassifier(base_classifier).fit(X_train, y_sl)
b30 = base_classifier.fit(X_train[u < LABELED_PROP], y_train[u < LABELED_PROP])

pred_s30 = s30.predict(X_test)
pred_b30 = b30.predict(X_test)

print(f"Benchmark accuracy is {100*accuracy_score(pred_b30, y_test):0.1f}")
print(f"Self-training accuracy is {100*accuracy_score(pred_s30, y_test):0.1f}")
