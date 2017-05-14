import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# 把data分割一部分訓練用一部分測試
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

# training
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# 測試訓練的模型
print(knn.predict(X_test))
print(y_test)


print(knn.score(X_test, y_test))

