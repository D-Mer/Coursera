import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import *
from functions import *

np.random.seed(1)

def different_hidden_size():
	X, Y = load_planar_dataset()
	hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
	for i, n_h in enumerate(hidden_layer_sizes):
		params, costs = nn_model(X, Y, n_h, num_iterations=10000, to_print=False)
		predictions = predict(params, X)
		result = np.zeros(Y.shape)
		result[predictions == Y] = 1
		print("hidden layer of size %d" % n_h)
		print("accuracy: %f" % float(np.count_nonzero(result) / Y.shape[1]))
		plot_decision_boundary(lambda x: predict(params, x.T), X, Y)
		plt.title("Decision Boundary for hidden layer size " + str(n_h))
		plt.show()

if __name__ == '__main__':
	sets = [noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure] = load_extra_datasets()
	for s in sets:
		X, Y = s[0].T, s[1].reshape(1, s[1].shape[0])
		n_h = 5
		plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
		plt.show()
		params, costs = nn_model(X, Y, n_h, num_iterations=5000, to_print=False)
		predictions = predict(params, X)
		result = np.zeros(Y.shape)
		result[predictions == Y] = 1
		print("accuracy: %f" % float(np.count_nonzero(result) / Y.shape[1]))
		plot_decision_boundary(lambda x: predict(params, x.T), X, Y)
		plt.title("Decision Boundary for hidden layer size " + str(n_h))
		plt.show()
