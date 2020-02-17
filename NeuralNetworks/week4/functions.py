import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward


def layer_sizes(X, Y):
	n_x = X.shape[0]
	n_h = 4  # neural num in hidden layers
	n_y = Y.shape[0]
	return n_x, n_h, n_y


def initialize_params(n_x, n_h, n_y):
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


def initialize_params_deep(layer_dims):
	np.random.seed(3)
	params = {}
	L = len(layer_dims)
	for l in range(1, L):
		params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
		params["b" + str(l)] = np.zeros((layer_dims[l], 1))
	return params


def linear_forward(A, W, b):
	Z = np.dot(W, A) + b
	assert Z.shape == (W.shape[0], A.shape[1])
	cache = (A, W, b)
	return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
	Z, linear_cache = linear_forward(A_prev, W, b)
	if activation == "sigmoid":
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		A, activation_cache = relu(Z)
	else:
		print("no such activation function: " + activation)
		return
	# cache = ((A[l-1],W[l],b[l]), Z[l])
	cache = (linear_cache, activation_cache)
	# A[l]
	return A, cache


def L_model_forward(X, params):
	caches = []
	A = X
	L = len(params) // 2  # params is the map of "Wn" and "bn", so here we divide it by 2
	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, params["W" + str(l)], params["b" + str(l)], "relu")
		caches.append(cache)
	AL, cache = linear_activation_forward(A, params["W" + str(L)], params["b" + str(L)], "sigmoid")
	caches.append(cache)
	assert AL.shape == (1, X.shape[1])
	assert len(caches) == L
	return AL, caches


def compute_cost(AL, Y):
	m = Y.shape[1]
	cost = - np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
	cost = np.squeeze(cost)
	assert cost.shape == ()
	return cost


def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]
	assert A_prev.shape[1] == dZ.shape[1]
	dW = np.dot(dZ, A_prev.T) / m
	db = np.sum(dZ, axis=1, keepdims=True) / m
	dA_prev = np.dot(W.T, dZ)
	return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation="relu"):
	linear_cache, activation_cache = cache
	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
	else:
		print("no such activation function: " + activation)
		return
	dA_prev, dW, db = linear_backward(dZ, linear_cache)
	return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
	grads = {}
	L = len(caches)
	m = AL.shape[1]
	assert Y.shape == AL.shape
	dAL = -((Y / AL) - (1 - Y) / (1 - AL))

	current_cache = caches[L - 1]
	# in CSDN blog, here it is "'dA'+str(L)" instead of "'dA'+str(L-1)",
	# but it use the wrong subscript in the following,
	# so that it still work ok.
	# Here I use the right subscript and all over the following
	grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = \
		linear_activation_backward(dAL, current_cache, "sigmoid")

	for l in reversed(range(1, L)):
		current_cache = caches[l - 1]
		dA_prev, dW_l, db_l = linear_activation_backward(grads["dA" + str(l)], current_cache, "relu")
		grads["dA" + str(l - 1)] = dA_prev
		grads["dW" + str(l)] = dW_l
		grads["db" + str(l)] = db_l

	return grads


def update_params(params, grades, alpha):
	L = len(params) // 2
	for l in range(L):
		params["W" + str(l + 1)] = params["W" + str(l + 1)] - alpha * grades["dW" + str(l + 1)]
		params["b" + str(l + 1)] = params["b" + str(l + 1)] - alpha * grades["db" + str(l + 1)]
	return params


def two_layer_model(X, Y, layer_dims, alpha=0.0075, num_iterations=3000, to_print=False, to_plot=True):
	np.random.seed(1)
	grads = {}
	costs = []
	n_x, n_h, n_y = layer_dims
	params = initialize_params(n_x, n_h, n_y)
	W1 = params["W1"]
	b1 = params["b1"]
	W2 = params["W2"]
	b2 = params["b2"]

	for i in range(num_iterations):
		A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
		A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
		cost = compute_cost(A2, Y)
		dA2 = -(Y / A2 - (1 - Y) / (1 - A2))
		dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
		dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
		grads["dW1"] = dW1
		grads["dW2"] = dW2
		grads["db1"] = db1
		grads["db2"] = db2
		params = update_params(params, grads, alpha)
		W1 = params["W1"]
		b1 = params["b1"]
		W2 = params["W2"]
		b2 = params["b2"]
		if i % 100 == 0:
			costs.append(cost)
			if to_print:
				print("times: ", i, "\t, cost: ", np.squeeze(cost))
	if to_plot:
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations/100')
		plt.title("Learning rate =" + str(alpha))
		plt.show()
	return params


def L_layer_model(X, Y, layer_dims, alpha=0.0075, num_iterations=3000, to_print=False, to_plot=True):
	np.random.seed(1)
	m = X.shape[1]
	params = initialize_params_deep(layer_dims)
	costs = []
	L = len(layer_dims)
	for i in range(num_iterations):
		AL, caches = L_model_forward(X, params)
		cost = compute_cost(AL, Y)
		grads = L_model_backward(AL, Y, caches)
		params = update_params(params, grads, alpha)
		if i % 100 == 0:
			costs.append(cost)
			if to_print:
				print("times: ", i, "\tcosts: ", cost)
	if to_plot:
		plt.plot(np.squeeze(costs))
		plt.ylabel("cost")
		plt.xlabel("iterations/100")
		plt.title("alpha = " + str(alpha))
		plt.show()
	return params


def predict(X, Y, params):
	m = X.shape[1]
	L = len(params) // 2
	y_hat = np.zeros(Y.shape)
	probas, caches = L_model_forward(X, params)
	y_hat[probas > 0.5] = 1
	print("accuracy: %f" % float(np.sum(y_hat == Y) / m))
	return y_hat


def print_mislabeled_images(classes, X, y, p):
	"""
	绘制预测和实际不同的图像。
		X - 数据集
		y - 实际的标签
		p - 预测
	"""
	a = p + y
	mislabeled_indices = np.asarray(np.where(a == 1))
	plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
	num_images = len(mislabeled_indices[0])
	for i in range(num_images):
		index = mislabeled_indices[1][i]

		plt.subplot(2, num_images, i + 1)
		plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
		plt.axis('off')
		plt.title(
			"Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
				"utf-8"))
		plt.show()


def predict_my_img(image, label, params):
	fname = "images/" + image
	image = np.array(imageio.imread(fname))
	num_px = 64
	image = np.array(Image.fromarray(image).resize((num_px, num_px)))
	my_image = image.reshape(num_px * num_px * 3, 1)
	my_predicted_image = predict(my_image, label, params)
	plt.imshow(image)
	plt.show()
	print("y_hat = ", my_predicted_image, "and y = ", label)
