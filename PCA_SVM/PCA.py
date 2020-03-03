#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:29:18 2020

@author: noel

Principal Component Analysis applied to the Iris dataset.

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier  # import class
#from sklearn.cross_validation import cross_val_score  # Deprecated
from sklearn.model_selection import cross_val_score

from sklearn import decomposition
from sklearn import datasets

# Load in the data for iris clasiffication
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# KNN with the original iris
# We have done our own knn algorithm using mode and euclidian distance.
knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()

#############################
### PCA with 2 components  ##
#############################
# In a sense, PCA is a kind of matrix factorization, since it decomposes a 
# matrix X into WEW^T. However, matrix factorization is a very general term.
# from https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca
np.random.seed(10)
l = np.random.choice(range(iris.data.shape[0]), iris.data.shape[0], replace=False)

X = iris.data[l][:4]
y = iris.target[l][:4]
# axis = 0 gives the mean along columns, axis = 1 gives mean along rows
X -= np.mean(X, axis=0)
cov_mat1 = np.dot(X.T, X)/(X.shape[0]-1)
# Careful, removing the parenthesis in the denomnator changes the order of the operations
# and it gives the wroing result..  
np.dot(X.T, X)/X.shape[0]-1
# Or
X = iris.data[l][:4]
y = iris.target[l][:4]
# rowvar = False means columns represents variables, otherwise row represent variables
cov_mat2 = np.cov(X, rowvar=False)
evals, evecs = np.linalg.eigh(cov_mat2)

# Geting the covariance matrix mack from eigen values (evals) and eigen vectors (evect)
M1 = np.matmul(evecs, np.diag(evals))
M2 = np.matmul(M1, evecs.T)
# M2 is now equal to cov_mat

# we now want to check how much of the covariance matrix is retained by each
# eigen value.
# Sort eigen values and return the index locations in the array
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]
evals = evals[idx]
variance_retained=np.cumsum(evals)/np.sum(evals)

# with sklearn we get the equivalent results but the results are arranged different
# because of a small difference in algorith with numpy
pca = PCA()
X_transformed = pca.fit_transform(X)
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / X.shape[0]-1
pca.explained_variance_
evals
pca.components_.T
evecs
# PCA is a statistical procedure that uses an orthogonal transformation to 
# convert a set of observations of possibly correlated variables (entities 
# each of which takes on various numerical values) into a set of values of 
# linearly uncorrelated variables called principal components.

# The question boils down to whether you what to subtract the means and divide 
# by standard deviation first. 
# WARNING: The explanation above is oversimplified. Do not confuse PCA and the 
#          related SVD.
X = iris.data[l]
y = iris.target[l]

# fit_transform(self, X, y=None)
# Fit the model with X and apply the dimensionality reduction on X.
X_r2 = pca.fit_transform(X)

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA(2 components) of IRIS dataset')

X_transformedSK = pca.transform(X)
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_transformedSK[y == i, 0], X_transformedSK[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('PCA(2 components) of IRIS dataset')
# Apply dimensionality reduction to X.
# X is projected on the first principal components previously extracted from a training set.
# only 2 columns!!

# KNN with PCAed data
knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X_transformedSK, y, cv=10, scoring='accuracy').mean()

X_reconstituted = pca.inverse_transform(X_transformedSK)
# Turn it back into its 4 column using only 2 principal components

plt.scatter(X[:,2], X[:,3])
plt.scatter(X_reconstituted[:,2], X_reconstituted[:,3])
# it is only looking at 2 dimensions of data!
#############################
### PCA with 1 components  ##
#############################
plt.cla()
pca = decomposition.PCA(n_components=1)
pca.fit(X)
X_1 = pca.transform(X)

X_1

# KNN with 3 components
knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X_1, y, cv=10, scoring='accuracy').mean()

X_reconstituted1 = pca.inverse_transform(X_1)

plt.scatter(X[:,2], X[:,3])
plt.scatter(X_reconstituted1[:,2], X_reconstituted1[:,3])
#plt.scatter(X[:,2], X_reconstituted[:,2])
#plt.scatter(X[:,3], X_reconstituted[:,3])
#############################
### PCA with 2 components  ##
#############################
#plt.cla()
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_2 = pca.transform(X)

X_2

# KNN with 3 components
knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X_2, y, cv=10, scoring='accuracy').mean()

X_reconstituted2 = pca.inverse_transform(X_2)

plt.scatter(X[:,2], X[:,3])
plt.scatter(X_reconstituted2[:,2], X_reconstituted2[:,3])
#############################
### PCA with 3 components  ##
#############################
#plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X_3 = pca.transform(X)

X_3

# KNN with 3 components
knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X_3, y, cv=10, scoring='accuracy').mean()

X_reconstituted3 = pca.inverse_transform(X_3)

plt.scatter(X[:,2], X[:,3])
plt.scatter(X_reconstituted3[:,2], X_reconstituted3[:,3])
#############################
### choosing components  ####
#############################
pca = decomposition.PCA(n_components=4)
X_4 = pca.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5)
cross_val_score(knn, X_4, y, cv=10, scoring='accuracy').mean()

X_reconstituted4 = pca.inverse_transform(X_4)

plt.scatter(X[:,2], X[:,3])
plt.scatter(X_reconstituted4[:,2], X_reconstituted4[:,3])
#############################
### choosing components  ####
#############################
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.cla()
plt.plot(pca.explained_variance_ratio_)
plt.title('Variance explained by each principal component')
plt.ylabel(' % Variance Explained')
plt.xlabel('Principal component')
# 2 components is enough!!

fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5)
c = ax0.pcolor(X)
ax0.set_title('X')

c = ax1.pcolor(X_reconstituted1)
ax1.set_title('X(n=1)')
ax1.set_axis_off()

c = ax2.pcolor(X_reconstituted2)
ax2.set_title('X(n=2)')
ax2.set_axis_off()

c = ax3.pcolor(X_reconstituted3)
ax3.set_title('X(n=3)')
ax3.set_axis_off()

c = ax4.pcolor(X_reconstituted4)
ax4.set_title('X(n=4)')
ax4.set_axis_off()

fig.colorbar(c, ax=ax4)
#fig.tight_layout()
plt.show()

fig, (ax0, ax1) = plt.subplots(1, 2)
c = ax0.pcolor(X)
ax0.set_title('X')

c = ax1.pcolor(X_reconstituted1)
ax1.set_title('X(n=1)')
fig.tight_layout()
plt.show()

print('Difference X x(n=1)',sum(sum(X-X_reconstituted1)))
print('Difference X x(n=2)',sum(sum(X-X_reconstituted2)))

import pandas as pd
idxs = ['X', 'X(n=1)', 'X(n=2)', 'X(n=3)']

box_plots_df = pd.DataFrame()
box_plots_df['X'] = [sum(X[i]-X_reconstituted1[i]) for i in range(X.shape[0])]
box_plots_df['X(n=1)'] = [sum(X[i]-X_reconstituted2[i]) for i in range(X.shape[0])]
box_plots_df['X(n=2)'] = [sum(X[i]-X_reconstituted3[i]) for i in range(X.shape[0])]
box_plots_df['X(n=3)'] = [sum(X[i]-X_reconstituted4[i]) for i in range(X.shape[0])]
box_plots_df.boxplot(column=idxs)


