% open the csv file containing the data into a table.
filename = "all-data.csv";
T = readtable(filename);

% convert the table column to an array
sentences = table2array(T(:, 2));

% build the labels vector
labels = categorical(table2array(T(:, 1)));

% tokenize the sentences
documents = tokenizedDocument(sentences);

% Perform pre-processing on the sentences
documents = removeStopWords(documents);
documents = erasePunctuation(documents);
documents = addPartOfSpeechDetails(documents);
documents = normalizeWords(documents,'Style','lemma');

% building the Bag-Of-Words
bag = bagOfWords(documents);

% removing common stop words and words with fewer than 100 occurences
newBag = removeInfrequentWords(removeWords(bag, stopWords), 90);

% Building the full TF-IDF matrix for the resulting bag
M = tfidf(newBag);
M1 = full(M);

% define the objective function for Bayesian Optimization
fun = @(x)knnBayesObj(x, M1, labels);

% define the hyperparameter space for Bayesian Optimization
vars = optimizableVariable('k', [1, 10], 'Type', 'integer');

% perform Bayesian Optimization
results = bayesopt(fun, vars, 'Verbose', 1, 'AcquisitionFunctionName', 'expected-improvement-plus');

% extract the best hyperparameters
best_k = results.XAtMinObjective.k;

% train the KNN model on the full dataset using the best k value
knn_model = fitcknn(M1, labels, 'NumNeighbors', best_k);

% make predictions on the test set
test_features = M1(2423:end, :);
test_labels = labels(2423:end);
knn_predictions = predict(knn_model, test_features);

% calculate the accuracy of the model
accuracy = sum(knn_predictions == test_labels) / numel(test_labels);

% visualise the model using confusion chart
figure(1)
confusionchart(test_labels, knn_predictions, 'Title', 'K Nearest Neighbors confusion chart');

fprintf('Accuracy: %.2f%%\n', 100 * accuracy);
