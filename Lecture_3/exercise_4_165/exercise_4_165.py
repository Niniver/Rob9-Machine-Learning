# ###################################
# Group ID : <165>
# Members : <Emil Frydenholm, Mads Wenneberg Mikkelsen>
# Date : <23-09-2035>
# Lecture: <4> <Dimensionality reduction>
# Dependencies: numpy, SciKit-learn
# Python version:3.11.9
# Functionality:This code loads selected MNIST digit data (5, 6, 8), reduces its dimensionality to 2D using LDA and PCA, fits Gaussian classifiers, and visualizes both classification accuracy and decision boundaries.
# ###################################
# %%
import numpy as np
from scipy.stats import multivariate_normal as norm
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

# -----------------------
# Load Data
# -----------------------
train5 = np.loadtxt("mnist_all_ASCII/mnist_all/train5.txt") / 255
train6 = np.loadtxt("mnist_all_ASCII/mnist_all/train6.txt") / 255
train8 = np.loadtxt("mnist_all_ASCII/mnist_all/train8.txt") / 255

train5_target = 5 * np.ones(len(train5))
train6_target = 6 * np.ones(len(train6))
train8_target = 8 * np.ones(len(train8))

train_data = np.concatenate([train5, train6, train8])
train_targets = np.concatenate([train5_target, train6_target, train8_target])

test5 = np.loadtxt("mnist_all_ASCII/mnist_all/test5.txt") / 255
test6 = np.loadtxt("mnist_all_ASCII/mnist_all/test6.txt") / 255
test8 = np.loadtxt("mnist_all_ASCII/mnist_all/test8.txt") / 255

test5_target = 5 * np.ones(len(test5))
test6_target = 6 * np.ones(len(test6))
test8_target = 8 * np.ones(len(test8))

test_data = np.concatenate([test5, test6, test8])
test_targets = np.concatenate([test5_target, test6_target, test8_target])

classes = np.array([5, 6, 8])

# -----------------------
# Helper: Gaussian classifier
# -----------------------
def fit_gaussians(train_data_2d, train_targets, classes):
    means, covs, priors = {}, {}, {}
    for c in classes:
        data_c = train_data_2d[train_targets == c]
        means[c] = np.mean(data_c, axis=0)
        covs[c] = np.cov(data_c, rowvar=False)
        priors[c] = len(data_c) / len(train_data_2d)
    return means, covs, priors

def predict_gaussian(test_data_2d, means, covs, priors, classes):
    likelihoods = np.zeros((len(test_data_2d), len(classes)))
    for i, c in enumerate(classes):
        rv = norm(mean=means[c], cov=covs[c])
        likelihoods[:, i] = rv.pdf(test_data_2d) * priors[c]
    preds = classes[np.argmax(likelihoods, axis=1)]
    return preds

def plot_decision_boundary(train_data_2d, train_targets, means, covs, priors, classes, title):
    # Define grid
    x_min, x_max = train_data_2d[:, 0].min() - 1, train_data_2d[:, 0].max() + 1
    y_min, y_max = train_data_2d[:, 1].min() - 1, train_data_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict on grid
    preds = predict_gaussian(grid, means, covs, priors, classes)
    Z = preds.reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=len(classes))
    for c in classes:
        plt.scatter(train_data_2d[train_targets == c, 0],
                    train_data_2d[train_targets == c, 1],
                    label=f"Class {int(c)}", alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.show()

# -----------------------
# LDA
# -----------------------
lda = LDA(n_components=2)
train_data_lda = lda.fit_transform(train_data, train_targets)
test_data_lda = lda.transform(test_data)

means_lda, covs_lda, priors_lda = fit_gaussians(train_data_lda, train_targets, classes)
predictions_lda = predict_gaussian(test_data_lda, means_lda, covs_lda, priors_lda, classes)
accuracy_lda = np.mean(predictions_lda == test_targets)
print(f"LDA Gaussian Classifier Accuracy: {accuracy_lda:.3f}")

plot_decision_boundary(train_data_lda, train_targets, means_lda, covs_lda, priors_lda, classes,
                       title="LDA Decision Boundaries")

# -----------------------
# PCA
# -----------------------
pca = PCA(n_components=2)
train_data_pca = pca.fit_transform(train_data)
test_data_pca = pca.transform(test_data)

means_pca, covs_pca, priors_pca = fit_gaussians(train_data_pca, train_targets, classes)
predictions_pca = predict_gaussian(test_data_pca, means_pca, covs_pca, priors_pca, classes)
accuracy_pca = np.mean(predictions_pca == test_targets)
print(f"PCA Gaussian Classifier Accuracy: {accuracy_pca:.3f}")

plot_decision_boundary(train_data_pca, train_targets, means_pca, covs_pca, priors_pca, classes,
                       title="PCA Decision Boundaries")
