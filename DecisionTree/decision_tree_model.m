%open the csv file containing the data into a table.
filename = "all-data.csv";
T = readtable('all-data.csv');

%convert the table column to an array
sentences = table2array(T(:, 2));

%build the labels vector
labels = unique(table2array(T(:, 1)));

%tokenize the sentences2
%documents = tokenizedDocument(sentences);

% Perform pre-processing on the sentences
documents = tokenizedDocument(sentences);
documents = removeStopWords(documents);
documents = erasePunctuation(documents);
documents = addPartOfSpeechDetails(documents);
documents = normalizeWords(documents,'Style','lemma');

%building the Bag-Of-Words
bag = bagOfWords(documents);

%removing common stop words and
% words with fewer than 100 occurences
newBag = removeInfrequentWords(removeWords(bag, stopWords), 90);


%Building the full TF-IDF matrix for the resulting bag.
M = tfidf(newBag);
M1 = array2table(full(M));

%creating the features matrix for training
training_features = M1(1:2423, :);
testing_features = M1(2423:end, :);

%creating the labels vector
training_labels = T(1:2423, 1);
testing_labels = table2array(T(2423:end, 1));

% Train decision tree models with different values of MaxNumSplits
tree_model3 = fitctree(training_features, training_labels, 'MaxNumSplits', 6);
tree_predictions3 = predict(tree_model3, testing_features);
accuracyDTree = sum(strcmp(tree_predictions3, testing_labels)) / numel(testing_labels);


figure(6)
confusionchart(testing_labels, tree_predictions3, 'Title', 'Decision Tree confusion chart');
% Compare accuracies of SVM and decision tree models
fprintf('Decision Tree Accuracy: %.2f%%\n', 100 * accuracyDTree);