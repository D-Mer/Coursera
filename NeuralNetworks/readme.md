# My learning of Neural Network

I learned this course in Bilibili instead of Coursera in that I was f**ked by the network when learning in Coursera. Luckily I found there are videos in Bilibili with double subtitles in English and Chinese.(Maybe it's just my stupid) And I found there are exercises and other material recorded by others, which I had downloaded to the folds named "week x".



Course videos: https://www.bilibili.com/video/av66314465

Exercises: https://blog.csdn.net/u013733326/article/details/79827273



# Exercises

## week 1

completed time: 2020/2/11

## week 2

completed time: 2020/2/13

## week 3

completed time: 2020/2/14

**note**: since here we should change the function "plot_decision_boundary" in file "planar_utils.py".

Change the sentence **"plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)"**  to **"plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)"**. 

Otherwise the program will report error : "ValueError: 'c' argument has 1 elements, which is not acceptable for use with 'x' with size 400, 'y' with size 400."

And remember that we should change sentences like that anywhere from now on.

**An interesting point**: in cost function "compute_cost()", there may be zero element in A2, which will cause the program report error: "divide by zero encountered in log: log_temp = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)". In test, the cost will be "-inf" after about 47 loop time. It may be gradient disappeared problem. I guess we can use ReLU to solve the problem(but I didn't try that)



## week 4

completed time: 2020/2/15

**note**: in function "initialize_params()", we use "\*0.01" to initialize Wn in that it's a shallow nework. In function "initialize_params_deep()", we use "/np.sqrt(layers_dims[i - 1])" to initialize Wn instead of "\*0.01" in that it will affect the learning rate in deep neural network. It makes the deep network learn very slow. I tried that at first and found the cost was staying in 0.64 but actually decreasing, while in shallow network we can almost ignore it, for which we can just use "\*0.01".

And we can know why and how it affects in the Next Course of the specialization