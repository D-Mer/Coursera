import numpy as np


def sigmoid(x):
	s = 1 / (1 + np.exp(-x))
	return s


def layer_sizes(X, Y):
	n_x = X.shape[0]
	n_h = 4  # neural num in hidden layers
	n_y = Y.shape[0]
	return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
	np.random.seed(2)
	W1 = np.random.randn(n_h, n_x) * 0.01
	b1 = np.zeros(shape=(n_h, 1))
	W2 = np.random.randn(n_y, n_h) * 0.01
	b2 = np.zeros(shape=(n_y, 1))
	params = {
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2
	}
	return params


def forward_propagation(X, params):
	W1 = params["W1"]
	b1 = params["b1"]
	W2 = params["W2"]
	b2 = params["b2"]
	assert W1.shape[0] == b1.shape[0]
	assert W1.shape[1] == X.shape[0]
	assert W1.shape[0] == W2.shape[1]
	assert W2.shape[0] == b2.shape[0]

	Z1 = np.dot(W1, X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)
	assert A2.shape == (1, X.shape[1])

	cache = {
		"Z1": Z1,
		"A1": A1,
		"Z2": Z2,
		"A2": A2
	}
	return A2, cache


def compute_cost(A2, Y, params=None):  # params are unused in CSDN blog, too, so I keep it here for debug later
	m = Y.shape[1]
	log_temp = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
	cost = -np.sum(log_temp) / m
	cost = float(np.squeeze(cost))
	assert (isinstance(cost, float))
	return cost


def backward_propagation(params, cache, X, Y):
	m = X.shape[1]
	W1 = params["W1"]
	W2 = params["W2"]
	b1 = params["b1"]
	b2 = params["b2"]
	A1 = cache["A1"]
	Z1 = cache["Z1"]
	A2 = cache["A2"]
	Z2 = cache["Z2"]

	assert Z2.shape == A2.shape == Y.shape
	dZ2 = A2 - Y
	dW2 = np.dot(dZ2, A1.T) / m
	db2 = np.sum(dZ2, axis=1, keepdims=True) / m
	dZ1 = np.dot(W2.T, dZ2) * (1 - A1 ** 2)
	dW1 = np.dot(dZ1, X.T) / m
	db1 = np.sum(dZ1, axis=1, keepdims=True) / m
	grads = {
		"dW1": dW1,
		"dW2": dW2,
		"dZ1": dZ1,
		"dZ2": dZ2,
		"db1": db1,
		"db2": db2
	}
	return grads


def update_params(params, grads, alpha=1.2):
	W1, W2 = params["W1"], params["W2"]
	b1, b2 = params["b1"], params["b2"]
	dW1, dW2 = grads["dW1"], grads["dW2"]
	db1, db2 = grads["db1"], grads["db2"]

	W1 = W1 - alpha * dW1
	b1 = b1 - alpha * db1
	W2 = W2 - alpha * dW2
	b2 = b2 - alpha * db2

	params = {
		"W1": W1,
		"b1": b1,
		"W2": W2,
		"b2": b2
	}
	return params


def nn_model(X, Y, n_h, num_iterations, to_print=False):
	np.random.seed(3)
	n_x = layer_sizes(X, Y)[0]
	# n_h = layer_size(X, Y)[1] was defined as constant 4, so we use the transmitted param
	n_y = layer_sizes(X, Y)[2]
	params = initialize_parameters(n_x, n_h, n_y)

	cost = 0
	costs = []
	for i in range(num_iterations):
		A2, cache = forward_propagation(X, params)
		cost = compute_cost(A2, Y, params)
		grads = backward_propagation(params, cache, X, Y)
		params = update_params(params, grads, alpha=1.2)
		if i % 1000 == 0:
			costs.append(cost)
			if to_print:
				print("times: ", i, "\tcost: ", str(cost))
	costs.append(cost)
	return params, costs


def predict(params, X):
	A2, cache = forward_propagation(X, params)
	predictions = np.round(A2)
	return predictions
