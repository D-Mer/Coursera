import numpy as np


def sigmoid(z):
	result = 1 / (1 + np.exp(-z))
	return result


def initialize_with_zeros(dim):
	w = np.zeros(shape=(dim, 1))
	b = 0
	return w, b


def propagate(w, b, X, Y):
	m = X.shape[1]
	A = sigmoid(np.dot(w.T, X) + b)
	cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

	dZ = A - Y
	dw = (1 / m) * np.dot(X, dZ.T)
	db = (1 / m) * np.sum(dZ)
	assert (dw.shape == w.shape)
	assert (db.dtype == float)

	return {"dw": dw, "db": db}, cost


def optimize(w, b, X, Y, num_iterations, alpha, print_cost=False):
	costs = []
	for i in range(num_iterations):
		grads, cost = propagate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]
		# note that -= can't be used here
		w = w - alpha * dw
		b = b - alpha * db
		if i % 100 == 0:
			costs.append(cost)
			if print_cost:
				print("time: " + str(i) + "\tcost: " + str(cost))

	params = {
		"w": w,
		"b": b
	}
	return params, grads, costs


def predict(w, b, X):
	m = X.shape[1]
	Y_predict = np.zeros((1, m))
	assert w.shape == (X.shape[0], 1)

	A = sigmoid(np.dot(w.T, X) + b)
	Y_predict[A > 0.5] = 1
	assert Y_predict.shape == (1, m)
	return Y_predict


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, alpha=0.5, to_print=False):
	w, b = initialize_with_zeros(X_train.shape[0])
	params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, alpha, to_print)
	w = params["w"]
	b = params["b"]
	if to_print:
		print("final parameters:")
		print(params)
	Y_predict_train = predict(w, b, X_train)
	Y_predict_test = predict(w, b, X_test)
	hit_rate_train = np.count_nonzero(Y_train == Y_predict_train)
	hit_rate_test = np.count_nonzero(Y_test == Y_predict_test)

	result = {
		"costs": costs,
		"w": w,
		"b": b,
		"learn_rate": alpha,
		"num_iterations": num_iterations,
		"Y_predict_train": Y_predict_train,
		"Y_predict_test": Y_predict_test,
		"hit_rate_train": hit_rate_train / Y_train.shape[1],
		"hit_rate_test": hit_rate_test / Y_test.shape[1]
	}

	return result


if __name__ == '__main__':
	print("sigmoid test:")
	print(sigmoid(0))
	print(sigmoid(9.2))
	print()
	print("propagate test:")
	w = np.array([[1], [2]])
	b = 2
	X = np.array([[1, 2], [3, 4]])
	Y = np.array([[1, 0]])
	print(Y.shape)
	grads, cost = propagate(w, b, X, Y)
	print(grads, cost)
	print()
	print("optimize test:")
	params, grades, costs = optimize(w, b, X, Y, 200, 0.009, False)
	print(params)

	print()
	w = params["w"]
	b = params["b"]
	print("predict test:")
	print(predict(w, b, X))
