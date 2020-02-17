import numpy as np
import matplotlib.pyplot as plt
import h5py
import pylab
from lr_utils import load_dataset
import Functions

if __name__ == '__main__':
	# load dataset
	train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

	# preview a sample of train set
	# index = 25
	# plt.imshow(train_set_x_orig[index])
	# plt.show()

	m_train = train_set_y.shape[1]
	m_test = test_set_y.shape[1]
	num_px = train_set_x_orig.shape[1]

	# flatten the images
	train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
	test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

	# regularize the images pixes
	train_set_x = train_set_x_flatten / 255
	test_set_x = test_set_x_flatten / 255

	learning_rates = [0.01, 0.001, 0.0001]
	results = {}
	for i in learning_rates:
		result = (Functions.model(train_set_x, train_set_y, test_set_x, test_set_y, 2000, i))
		print("learning_rate = " + str(i))
		print("w: " + str(result["w"]))
		print("b: " + str(result["b"]))
		print("hit_rate_train: " + str(result["hit_rate_train"]))
		print("hit_rate_test: " + str(result["hit_rate_test"]))
		results[str(i)] = result

		costs = np.squeeze(result["costs"])
		plt.plot(costs, label=str(results[str(i)]["learn_rate"]))

	plt.ylabel("costs")
	plt.xlabel("iterations/100")
	legend = plt.legend(loc='upper center', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	plt.show()
