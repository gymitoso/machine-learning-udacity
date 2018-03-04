# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Introdução do problema - https://github.com/udacity/machine-learning/blob/master/projects/practice_projects/naive_bayes_tutorial/Bayesian_Inference.ipynb

# Dataset obtido em - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

df['label'] = df.label.map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

# Instancia o método CountVectorizer
count_vector = CountVectorizer()

# Ajusta os dados de treinamento e retorna a matriz
training_data = count_vector.fit_transform(X_train)

# Transforma dados de teste e retorna a matriz
testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

# Resultados
print('Acurácia: ', format(accuracy_score(y_test, predictions)))
print('Precisão: ', format(precision_score(y_test, predictions)))
print('Recall: ', format(recall_score(y_test, predictions)))
print('Escore F1: ', format(f1_score(y_test, predictions)))
