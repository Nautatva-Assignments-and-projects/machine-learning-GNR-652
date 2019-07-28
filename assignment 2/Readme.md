So here you need to implement SVM for credit-card fraud...There is a kaggle competition for the same..

https://www.kaggle.com/pierra/credit-card-dataset-svm-classification

Use this data..And only use the built-in function for convex optimization.


1. The dataset is for credit card fraud. It is a binary classification task (labels 0 and 1). But recall as in SVM, we prefer the labels to be +1 and -1...So, convert the labels accordingly. Note that the dataset is highly biased. You may consider 100 +1 and 100 -1 samples randomly.

2. In the dual formulation based SVM, we convert the primal Lagrangian to the dual formulation which is a convex quadratic optimization problem with linear constraints. We apply a convex optimization function (available in matlab/python) to solve the \alpha values (the lagrangian multipliers). Note that if there are N data points in the training set, \alpha will be a vector of 1 x N (corresponding to the points). As we discussed, many of these N \alpha values will be 0 and the ones corresponding to support vector will be non-zero.

Also note that we converted the dual formula in the matrix notation. Use that.

3. One popular function class in SVM for convex optimisation is "cvxopt.solvers.qp"...It requires a particular form for the optimisation problem (see the attached do). So convert the matrix based dual formula into the one "cvxopt" accepts.

4. It will then do the optimization itself and will return the \alpha.

5. You can identify the support vectors now. Calculate "b" using the standard form (there in the ppt)

6. Take a test set and check the class labels with sgn(wx+b). (w = \sum \alpha_i y_i x_i)

7. Note that here we are solving the standard linear SVM, so no kernel function (or we use the linear dot product kernel).

8. Check the testing accuracy based on the % of correctly classified samples in the test set. 

Finally: So basically, you have to process the data and make in into the format acceptable by cvxopt. Then solve for \alpha. Then calculate "b". Then use the classifier form of Step-6 for testing.



So regarding the submission:



1. Randomly select 5 pairs of training and test sets (100 samples each).

2. Train and report the accuracies.



(Dataset link: https://www.kaggle.com/pierra/credit-card-dataset-svm-classification)



You have to submit the source file and a doc reporting the results in a zip file.

Note: hard margin only. You may experiment on the features,..its upto you.....