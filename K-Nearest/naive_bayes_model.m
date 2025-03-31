function accuracy = knnBayesObj(params, M1, labels)
%KNNBAYESOBJ Objective function for k-nearest neighbor classification with Bayesian optimization

% Convert the parameter values to integer
k = round(params.k);

% Train the KNN model with the given k value
knn_model = fitcknn(M1, labels, 'NumNeighbors', k);

% Perform 5-fold cross-validation and compute the classification accuracy
cv = crossval(knn_model, 'KFold', 5);
accuracy = 1 - kfoldLoss(cv, 'LossFun', 'ClassifError');
end
