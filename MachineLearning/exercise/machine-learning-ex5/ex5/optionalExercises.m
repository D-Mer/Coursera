% This file is written by myself, using some codes in ex5.m

%====================================================================
% These codes are used to compute the error of test set, which isn't 
% used in ex5.m.
clear ; close all; clc
load ('ex5data1.mat');
m = size(X, 1);
p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

theta = trainLinearReg(X_poly,y,3);

J = linearRegCostFunction(X_poly_test,ytest,theta,0);
fprintf('the cost is %f\n',J);


%====================================================================
% These codes are used to plotting learning curves
% with randomly selected examples

lambda = 0.01;
randomIndex = randperm(m);
X_random = X_poly(randomIndex,:);
y_random = y(randomIndex,:);
randomIndex = randperm(size(Xval,1));
X_random_val = X_poly_val(randomIndex,:);
y_random_val = yval(randomIndex,:);

[error_train, error_val] = ...
    learningCurve([ones(m, 1) X_random], y_random, ...
                  [ones(size(X_random_val, 1), 1) X_random_val], y_random_val, ...
                  lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])