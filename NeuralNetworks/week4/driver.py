import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
import lr_utils
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from functions import *
np.random.seed(1)

if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_dataset()

    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_x = train_x_flatten / 255
    train_y = train_set_y
    test_x = test_x_flatten / 255
    test_y = test_set_y

    layers_dims = [12288, 20, 7, 5, 1]  # 5-layer model
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, to_print=True, to_plot=True)

    pred_train = predict(train_x, train_y, parameters)  # 训练集
    pred_test = predict(test_x, test_y, parameters)  # 测试集

    predict_my_img("my_image1.jpg", np.array([[1]]), parameters)
    predict_my_img("my_image2.jpg", np.array([[1]]), parameters)
    predict_my_img("my_image3.jpg", np.array([[0]]), parameters)

    # print_mislabeled_images(classes, test_x, test_y, pred_test)