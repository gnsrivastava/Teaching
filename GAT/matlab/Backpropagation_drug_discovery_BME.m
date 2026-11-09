%% 1. Environment Setup & Configuration
clear; clc; rng(42);

% Specify the path to your .mat dataset file
matFileName = 'BBBP.mat'; 

%% 2. Load Dataset directly from MAT-file
if ~isfile(matFileName)
    error('File "%s" not found. Please ensure the .mat file is in the current working directory.', matFileName);
end

fprintf('Loading dataset from %s...\n', matFileName);
dataStruct = load(matFileName);

% Automatically extract feature matrix (X) and binary labels
if isfield(dataStruct, 'X') && isfield(dataStruct, 'labels')
    X = single(dataStruct.X);          % Feature matrix [N x NumFeatures]
    labels = double(dataStruct.labels); % Binary targets [N x 1]
else
    error('The specified .mat file must contain variables named "X" and "labels".');
end

[numSamples, numFeatures] = size(X);
fprintf('Successfully loaded dataset:\n');
fprintf('  - Samples:  %d\n', numSamples);
fprintf('  - Features: %d\n', numFeatures);

%% 3. Manual Stratified Data Partitioning (No Statistics Toolbox Required)
posIdx = find(labels == 1);
negIdx = find(labels == 0);

% Shuffle class indices deterministically
posIdx = posIdx(randperm(length(posIdx)));
negIdx = negIdx(randperm(length(negIdx)));

% Split ratios: 80% Train, 10% Validation, 10% Test
splitClass = @(idx) struct(...
    'train', idx(1:round(0.80 * length(idx))), ...
    'valid', idx(round(0.80 * length(idx)) + 1 : round(0.90 * length(idx))), ...
    'test',  idx(round(0.90 * length(idx)) + 1 : end));

posSplit = splitClass(posIdx);
negSplit = splitClass(negIdx);

trainIdx = [posSplit.train; negSplit.train];
validIdx = [posSplit.valid; negSplit.valid];
testIdx  = [posSplit.test;  negSplit.test];

% Format inputs for trainNetwork
XTrain = X(trainIdx, :); yTrain = categorical(labels(trainIdx));
XValid = X(validIdx, :); yValid = categorical(labels(validIdx));
XTest  = X(testIdx, :);  yTest  = categorical(labels(testIdx));

fprintf('Split Summary -> Train: %d | Validation: %d | Test: %d\n', ...
    length(trainIdx), length(validIdx), length(testIdx));

%% 4. Define Neural Network Architecture (With Regularization & Dropout)
layers = [
    featureInputLayer(numFeatures, 'Name', 'input')
    
    fullyConnectedLayer(32, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.4, 'Name', 'drop1')          % High dropout rate for high-dim sparse input
    
    fullyConnectedLayer(16, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.3, 'Name', 'drop2')          % Dropout before output projection
    
    fullyConnectedLayer(2, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% 5. Configure Regularized Training Options
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'auto', ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 3e-4, ...               % Lower learning rate to avoid sharp spikes
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...              % Decay learning rate every 20 epochs
    'LearnRateDropFactor', 0.5, ...
    'L2Regularization', 1e-2, ...               % Stronger weight penalty to prevent memorization
    'ValidationData', {XValid, yValid}, ...
    'ValidationFrequency', 5, ...               % Frequent validation checks
    'ValidationPatience', 8, ...                % Strict early stopping
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% 6. Train Network (Backpropagation)
fprintf('\nTraining neural network \n');
net = trainNetwork(XTrain, yTrain, layers, options);

%% 7. Model Evaluation on Test Set
% 1. Predict class probability scores (Single output argument)
testScores = predict(net, XTest);
testProbs = testScores(:, 2); % Class '1' probability

% 2. Classify to get predicted categorical labels
testPreds = classify(net, XTest);

testTargets = double(yTest) - 1; % Convert categorical back to 0/1 numeric

% Native MATLAB ROC-AUC Calculation (without perfcurve)
[sortedProbs, sortOrder] = sort(testProbs, 'descend');
sortedTargets = testTargets(sortOrder);

tp = cumsum(sortedTargets == 1);
fp = cumsum(sortedTargets == 0);

testFPR = fp / sum(sortedTargets == 0);
testTPR = tp / sum(sortedTargets == 1);

testFPR = [0; testFPR];
testTPR = [0; testTPR];
testAUC = trapz(testFPR, testTPR); % Numerical integration for AUC

testAcc = mean(testPreds == yTest);

fprintf('\n--- Evaluation Metrics ---\n');
fprintf('Test ROC-AUC:  %.4f\n', testAUC);
fprintf('Test Accuracy: %.2f%%\n', testAcc * 100);

%% 8. Plot ROC Curve and Confusion Matrix
figure('Name', 'Model Evaluation', 'Position', [100, 100, 900, 400]);

% Subplot 1: ROC Curve
subplot(1, 2, 1);
plot(testFPR, testTPR, 'LineWidth', 2, 'Color', '#0072BD');
hold on;
plot([0 1], [0 1], '--k', 'LineWidth', 1);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Test ROC Curve (AUC = %.4f)', testAUC));
grid on;
axis square;

% Subplot 2: Static Confusion Matrix (Prevents Heatmap Linger Warning)
subplot(1, 2, 2);
cmMat = [sum(testTargets == 0 & double(testPreds)-1 == 0), sum(testTargets == 0 & double(testPreds)-1 == 1); ...
    sum(testTargets == 1 & double(testPreds)-1 == 0), sum(testTargets == 1 & double(testPreds)-1 == 1)];

imagesc(cmMat);
colormap(gca, flipud(bone)); % Crisp gray background
colorbar;
axis square;

% Annotate grid numbers explicitly
textStrings = num2str(cmMat(:), '%d');
textStrings = strtrim(cellstr(textStrings));
[xGrid, yGrid] = meshgrid(1:2, 1:2);
hStrings = text(xGrid(:), yGrid(:), textStrings(:), ...
    'HorizontalAlignment', 'center', ...
    'Color', 'red', 'FontSize', 14, 'FontWeight', 'bold');

set(gca, 'XTick', 1:2, 'XTickLabel', {'0', '1'}, ...
    'YTick', 1:2, 'YTickLabel', {'0', '1'}, ...
    'TickLength', [0 0]);
xlabel('Predicted Class');
ylabel('True Class');
title('Test Set Confusion Matrix');