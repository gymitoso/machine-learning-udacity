# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas
import numpy

# Dataset
data = pandas.read_csv('data.csv')

# Divide os dados em duas arrays para os classificadores
X = numpy.array(data[['x1', 'x2']])
y = numpy.array(data['y'])

# Divide em dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Logistic Regression (Regressão logística)
classifierLR = LogisticRegression()
classifierLR.fit(X_train,y_train)

predictions = classifierLR.predict(X_test)

# Resultados Logistic Regression
print('Logistic Regression')
print('Acurácia: ', format(accuracy_score(y_test, predictions)))
print('Precisão: ', format(precision_score(y_test, predictions)))
print('Recall: ', format(recall_score(y_test, predictions)))
print('Escore F1: ', format(f1_score(y_test, predictions)))


# Decision Tree (Árvore de decisão)
classifierDT = DecisionTreeClassifier()
classifierDT.fit(X,y)

predictions = classifierDT.predict(X_test)

# Resultados Decision Tree
print('Decision Tree')
print('Acurácia: ', format(accuracy_score(y_test, predictions)))
print('Precisão: ', format(precision_score(y_test, predictions)))
print('Recall: ', format(recall_score(y_test, predictions)))
print('Escore F1: ', format(f1_score(y_test, predictions)))

# Support Vector Machine (Máquina de vetores de suporte)
classifierSVM = SVC()
classifierSVM.fit(X,y)

predictions = classifierSVM.predict(X_test)

# Resultados Support Vector Machine
print('Support Vector Machine')
print('Acurácia: ', format(accuracy_score(y_test, predictions)))
print('Precisão: ', format(precision_score(y_test, predictions)))
print('Recall: ', format(recall_score(y_test, predictions)))
print('Escore F1: ', format(f1_score(y_test, predictions)))
