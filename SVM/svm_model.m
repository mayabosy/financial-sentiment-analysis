% Open the csv file containing the data into a table
filename = "all-data.csv";
T = readtable(filename);

% Convert the table column to an array
sentences = table2array(T(:, 2));

% Build the labels vector
labels = unique(table2array(T(:, 1)));

% Tokenize the sentences
documents = tokenizedDocument(sentences);
documents = removeStopWords(documents);
documents = erasePunctuation(documents);
documents = addPartOfSpeechDetails(documents);
documents = normalizeWords(documents,'Style','lemma');

% Build the Bag-Of-Words
bag = bagOfWords(documents);

% Remove common stop words and
% words with fewer than 100 occurrences
newBag = removeInfrequentWords(removeWords(bag, stopWords), 90);

% Build the full TF-IDF matrix for the resulting bag
M = tfidf(newBag);
M1 = array2table(full(M));

% Generate a random permutation of the indices
idx = randperm(size(M1, 1));

% Use the first 70% of the indices for training
train_idx = idx(1:floor(0.7 * numel(idx)));

% Use the remaining 30% of the indices for testing
test_idx = idx(floor(0.7 * numel(idx))+1:end);

% Create training and testing sets
training_features = M1(train_idx, :);
training_labels = T(train_idx, 1);
testing_features = M1(test_idx, :);
testing_labels = table2array(T(test_idx, 1)); % Convert to vector

% Train the SVM model on the training set
svmmodel = fitcecoc(training_features, training_labels);

% Predict the labels of the testing set using the trained SVM model
svm_predictions = predict(svmmodel, testing_features);

% Calculate the accuracy of the SVM model
num_correct_predictions_svm = sum(strcmp(svm_predictions, testing_labels));
svm_accuracy = num_correct_predictions_svm ./ numel(testing_labels);

% Plot the confusion chart
figure(4)
confusionchart(testing_labels, svm_predictions, 'Title', 'SVM confusion chart');

% Print the accuracy
fprintf('Accuracy: %.2f%%\n', 100 * svm_accuracy);
