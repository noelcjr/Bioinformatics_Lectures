#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:53:57 2020

@author: noel
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, linear_model, datasets
from sklearn.model_selection import cross_val_score
# new dataset, handwritten digits!
digits = datasets.load_digits()

digits.data
len(digits.data)      # 1,797 observations
len(digits.data[0])   # 8 x 8 pixel image

plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
# the number 9

digits.target
len(digits.target)
digits.target[-5]
# 9
digits_X, digits_y = digits.data, digits.target

# Let's try a SVM
clf = svm.SVC(gamma='auto')
clf.fit(digits_X,digits_y)
plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
clf.predict(digits.data)[-5]
# WOOOOO
cross_val_score(clf, digits_X, digits_y, cv=5, scoring='accuracy').mean()
# Guassian has two parameters, gamma and C
'''
Intuitively, the gamma parameter defines how far the influence of a 
single training example reaches, with low values meaning ‘far’ and 
high values meaning ‘close’. 

small gamma: the model is constrained, can under-fit!
big gamma: Tries to capture the shape too well: can over-fit!


small C: makes the decision surface smooth and simple, can under-fit!
big C: selects more support vectors: can over-fit!
'''
# note the scale of gamma and C
clf = svm.SVC(gamma=0.001, C=1)
cross_val_score(clf, digits_X, digits_y, cv=5, scoring='accuracy').mean()

# So we have gamma and C parameters that affect our results. It is important to explore a range
# of this values in order to find the optimal combination.
from sklearn.model_selection import GridSearchCV
clf = svm.SVC(C=1)
gamma_range = 10.**np.arange(-5, 2)
# C_range = 10.**np.arange(-2, 3)
param_grid = dict(gamma=gamma_range)
grid = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy', return_train_score=True)
grid.fit(digits_X, digits_y)

# check the results of the grid search
grid.cv_results_
grid.cv_results_.keys()
grid.cv_results_['mean_test_score']
grid.cv_results_['params']
grid.cv_results_['param_gamma']
# plot the results
import matplotlib.pyplot as plt
plt.plot(range(-5,2), grid.cv_results_['mean_test_score'])

# what was best?
grid.best_score_
grid.best_params_
grid.best_estimator_

# EXERCISE 
# add in a new parameter to search over, C
# gri search over values of 10^-5 to 10^2

# import some data to play with
iris = datasets.load_iris()
iris_X = iris.data[:, :2]  # we only take the first two features to avoid
                           # ugly slicing and print the two-dim dataset classification
iris_y = iris.target

clf = svm.SVC()
cross_val_score(clf, iris_X, iris_y, cv=5, scoring='accuracy').mean()
'''
Let's compare three SVMs with different kernels

Gaussian
Linear
Poly of degree 3
'''
# C is SVM regularization parameter
# https://datascience.stackexchange.com/questions/4943/intuition-for-the-regularization-parameter-in-svm
C = 1.0
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(iris_X, iris_y)  # default kernel
svc = svm.SVC(kernel='linear', C=C).fit(iris_X, iris_y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C, gamma='auto').fit(iris_X, iris_y)

# create a mesh to plot in
x_min, x_max = iris_X[:, 0].min() - 1, iris_X[:, 0].max() + 1
y_min, y_max = iris_X[:, 1].min() - 1, iris_X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))
# title for the plots
titles = ['SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

for i, clf in enumerate((svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    # Plot also the training points
    plt.scatter(iris_X[:, 0], iris_X[:, 1], c=iris_y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.show()

# MORE DATA to show advantages of SVM
from sklearn.datasets import make_circles
circles_X, circles_y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)
plt.scatter(circles_X[:,0], circles_X[:,1])

clf = svm.SVC(kernel = 'linear')        # I like lines
cross_val_score(clf, circles_X, circles_y, cv=5, scoring='accuracy').mean()

clf = svm.SVC(kernel = 'poly', degree = 3)        # I like 3rd degree polys
cross_val_score(clf, circles_X, circles_y, cv=5, scoring='accuracy').mean()

clf = svm.SVC(kernel = 'rbf')           # I like circles
cross_val_score(clf, circles_X, circles_y, cv=5, scoring='accuracy').mean()
# the radial basis function (rbf) fake projects the data into higher dimensions
# that accompany circles well. The image below show this.
import matplotlib.image as img 
im = img.imread('/home/noel/Projects/Lectures/PCA_SVM/rbf.png') 
# show image 
plt.imshow(im) 

# OK now with graphs :)
C = 1.0  # SVM regularization parameter
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(circles_X, circles_y)  # default kernel
svc = svm.SVC(kernel='linear', C=C).fit(circles_X, circles_y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(circles_X, circles_y)

# create a mesh to plot in
x_min, x_max = circles_X[:, 0].min() - 1, circles_X[:, 0].max() + 1
y_min, y_max = circles_X[:, 1].min() - 1, circles_X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))

# title for the plots
titles = ['SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

for i, clf in enumerate((svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    # Plot also the training points
    plt.scatter(circles_X[:, 0], circles_X[:, 1], c=circles_y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.show()
# a real thing of beauty

# Finally, Visualizing different C
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20
# figure number
fignum = 1
# fit the model
for name, penalty in (('unreg=1', 1), ('reg=0.05', 0.05)):
    clf = svm.SVC(kernel='linear', C=penalty)
    clf.fit(X, Y)
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    # plot the parallels to the separating hyperplane that pass through the
    # support vectors
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    yy_down = yy + a * margin
    yy_up = yy - a * margin
    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10)
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired)
    plt.axis('tight')
    plt.title(name)
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    fignum = fignum + 1
plt.show()






