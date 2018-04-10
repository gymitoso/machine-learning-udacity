""" lecture and example code for decision tree unit """

import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn.metrics import accuracy_score

from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

prettyPicture(clf, features_test, labels_test)

#### compute the accuracy on the testing data
def DTClassifier(min_split=2):
    clf = tree.DecisionTreeClassifier(min_samples_split=min_split)
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test,pred)
    return acc

acc_min_samples_split_2 = DTClassifier()
acc_min_samples_split_50 = DTClassifier(min_split=50)

print "Accuracy with min samples 2: ", format(round(acc_min_samples_split_2,3))
print "Accuracy with min samples 50: ", format(round(acc_min_samples_split_50,3))
