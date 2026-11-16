%% ========================================================================
%  BACKGROUND & DATASET OVERVIEW: Blood-Brain Barrier Penetration (BBBP)
%  ========================================================================
%  Biological Context:
%    The blood-brain barrier (BBB) is a selective semipermeable membrane that
%    protects the central nervous system (CNS) from circulating toxins while
%    regulating nutrient transport. Predicting whether a small-molecule drug
%    can cross the BBB is a critical step in designing CNS therapeutics (or
%    avoiding CNS side effects for peripheral drugs).
%
%  Dataset Description (MoleculeNet BBBP Benchmark):
%    - Source       : Martins et al. (2012) / MoleculeNet Benchmark Suite.
%    - Samples (N)  : ~2,039 chemical compounds with experimentally measured
%                     blood-brain barrier permeability.
%    - Features (X) : Precomputed molecular descriptors or structural 
%                     fingerprints (e.g., Morgan/ECFP fingerprints or physicochemical
%                     properties) stored as a [Samples x Features] matrix.
%    - Task Type    : Binary Classification:
%        * Class 1 (Permeable)     : Crosses the blood-brain barrier (logBB >= -1).
%        * Class 0 (Non-permeable) : Blocked by the blood-brain barrier (logBB < -1).
%
%  Model Architecture:
%    - Regularized Multi-Layer Perceptron (MLP) with ReLU activations, high
%      dropout rates, L2 weight decay, and early stopping to prevent overfitting
%      on high-dimensional molecular feature representations.
% ========================================================================

%% 1. Environment Setup & Configuration
% Clear workspace variables, reset command window, and set random seed for reproducibility
clear; clc; rng(42);

% Specify the path to your .mat dataset file
matFileName = 'BBBP.mat'; 

%% 2. Load Dataset directly from MAT-file
% Verify that the dataset file exists in the working directory before loading
if ~isfile(matFileName)
    error('File "%s" not found. Please ensure the .mat file is in the current working directory.', matFileName);
end

fprintf('Loading dataset from %s...\n', matFileName);
dataStruct = load(matFileName);

% Validate and extract feature matrix (X) and binary class labels
if isfield(dataStruct, 'X') && isfield(dataStruct, 'labels')
    X = single(dataStruct.X);           % Feature matrix [N x NumFeatures]
    labels = double(dataStruct.labels); % Binary targets [N x 1] (0 or 1)
else
    error('The specified .mat file must contain variables named "X" and "labels".');
end

% Extract dataset dimensions and print summary statistics
[numSamples, numFeatures] = size(X);
fprintf('Successfully loaded dataset:\n');
fprintf('  - Samples:  %d\n', numSamples);
fprintf('  - Features: %d\n', numFeatures);

%% 3. Manual Stratified Data Partitioning (No Statistics Toolbox Required)
% Separate indices of positive (permeable = 1) and negative (non-permeable = 0) compounds
posIdx = find(labels == 1);
negIdx = find(labels == 0);

% Randomly permute indices within each class using the fixed RNG seed
posIdx = posIdx(randperm(length(posIdx)));
negIdx = negIdx(randperm(length(negIdx)));

% Define helper anonymous function to split class indices into 80% Train / 10% Val / 10% Test
splitClass = @(idx) struct(...
    'train', idx(1:round(0.80 * length(idx))), ...
    'valid', idx(round(0.80 * length(idx)) + 1 : round(0.90 * length(idx))), ...
    'test',  idx(round(0.90 * length(idx)) + 1 : end));

% Apply stratified splitting independently to each class to maintain class balance
posSplit = splitClass(posIdx);
negSplit = splitClass(negIdx);

% Combine positive and negative indices for each data partition
trainIdx = [posSplit.train; negSplit.train];
validIdx = [posSplit.valid; negSplit.valid];
testIdx  = [posSplit.test;  negSplit.test];

% Subset feature matrices and convert numeric labels to categorical arrays for trainNetwork
XTrain = X(trainIdx, :); yTrain = categorical(labels(trainIdx));
XValid = X(validIdx, :); yValid = categorical(labels(validIdx));
XTest  = X(testIdx, :);  yTest  = categorical(labels(testIdx));

fprintf('Split Summary -> Train: %d | Validation: %d | Test: %d\n', ...
    length(trainIdx), length(validIdx), length(testIdx));

%% 4. Define Neural Network Architecture (With Regularization & Dropout)
% Construct a 2-hidden-layer feedforward neural network with dropout regularization
layers = [
    featureInputLayer(numFeatures, 'Name', 'input') % Input layer matching molecular feature count
    
    fullyConnectedLayer(32, 'Name', 'fc1')          % First hidden layer (32 units)
    reluLayer('Name', 'relu1')                      % Non-linear ReLU activation
    dropoutLayer(0.4, 'Name', 'drop1')              % 40% dropout to prevent overfitting on sparse features
    
    fullyConnectedLayer(16, 'Name', 'fc2')          % Second hidden layer (16 units)
    reluLayer('Name', 'relu2')                      % Non-linear ReLU activation
    dropoutLayer(0.3, 'Name', 'drop2')              % 30% dropout before final classification layer
    
    fullyConnectedLayer(2, 'Name', 'fc_out')        % Output logits for 2 classes (0 and 1)
    softmaxLayer('Name', 'softmax')                 % Convert logits into class probabilities
    classificationLayer('Name', 'output')           % Cross-entropy loss output layer
];

%% 5. Configure Regularized Training Options
% Specify Adam optimization settings, learning rate decay, L2 penalty, and early stopping
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'auto', ...             % Use GPU if available, otherwise CPU
    'MaxEpochs', 150, ...                           % Maximum number of training passes
    'MiniBatchSize', 64, ...                        % Number of compounds per gradient update
    'InitialLearnRate', 3e-4, ...                   % Conservative initial learning rate for stability
    'LearnRateSchedule', 'piecewise', ...           % Step-wise learning rate decay
    'LearnRateDropPeriod', 20, ...                  % Reduce learning rate every 20 epochs
    'LearnRateDropFactor', 0.5, ...                 % Halve the learning rate at each drop period
    'L2Regularization', 1e-2, ...                   % Strong L2 weight decay to curb overfitting
    'ValidationData', {XValid, yValid}, ...         % Held-out validation set for monitoring
    'ValidationFrequency', 5, ...                   % Evaluate validation loss every 5 iterations
    'ValidationPatience', 8, ...                    % Stop training if validation loss fails to improve 8 times
    'Shuffle', 'every-epoch', ...                   % Randomize training batch order each epoch
    'Plots', 'training-progress', ...               % Open interactive training progress window
    'Verbose', false);                              % Suppress command-line iteration spam

%% 6. Train Network (Backpropagation)
fprintf('\nTraining neural network \n');
net = trainNetwork(XTrain, yTrain, layers, options);

%% 7. Model Evaluation on Test Set
% 1. Predict posterior class probability scores [N_test x 2]
testScores = predict(net, XTest);
testProbs = testScores(:, 2); % Extract probability of positive class ('1')

% 2. Obtain discrete predicted class labels and convert ground-truth labels to numeric 0/1
testPreds = classify(net, XTest);
testTargets = double(yTest) - 1;

% Native MATLAB ROC-AUC Calculation (avoids Statistics and Machine Learning Toolbox dependency)
[sortedProbs, sortOrder] = sort(testProbs, 'descend'); % Sort predictions from highest to lowest confidence
sortedTargets = testTargets(sortOrder);                % Reorder ground-truth labels accordingly
tp = cumsum(sortedTargets == 1);                       % Cumulative true positives at each threshold
fp = cumsum(sortedTargets == 0);                       % Cumulative false positives at each threshold
testFPR = [0; fp / sum(sortedTargets == 0)];           % Normalize to False Positive Rate (prepend origin 0)
testTPR = [0; tp / sum(sortedTargets == 1)];           % Normalize to True Positive Rate (prepend origin 0)

% Compute Area Under the ROC Curve via trapezoidal numerical integration and overall accuracy
testAUC = trapz(testFPR, testTPR);
testAcc = mean(testPreds == yTest);

fprintf('\n--- Evaluation Metrics ---\n');
fprintf('Test ROC-AUC:  %.4f\n', testAUC);
fprintf('Test Accuracy: %.2f%%\n', testAcc * 100);

%% 8. Plot ROC Curve and Confusion Matrix
figure('Name', 'Model Evaluation', 'Position', [100, 100, 900, 400]);

% Subplot 1: Receiver Operating Characteristic (ROC) Curve
subplot(1, 2, 1);
plot(testFPR, testTPR, 'LineWidth', 2, 'Color', '#0072BD');
hold on;
plot([0 1], [0 1], '--k', 'LineWidth', 1); % Random classifier reference diagonal (AUC = 0.5)
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('Test ROC Curve (AUC = %.4f)', testAUC));
grid on;
axis square;

% Subplot 2: Static Confusion Matrix (uses imagesc to avoid heatmap linger warnings)
subplot(1, 2, 2);
% Manually populate 2x2 confusion matrix: rows = True Class (0, 1), cols = Predicted Class (0, 1)
cmMat = [sum(testTargets == 0 & double(testPreds)-1 == 0), sum(testTargets == 0 & double(testPreds)-1 == 1); ...
         sum(testTargets == 1 & double(testPreds)-1 == 0), sum(testTargets == 1 & double(testPreds)-1 == 1)];
imagesc(cmMat);
colormap(gca, flipud(bone)); % Light background for low counts, darker shading for high counts
colorbar;
axis square;

% Overlay exact sample counts inside each cell of the 2x2 matrix
textStrings = num2str(cmMat(:), '%d');
textStrings = strtrim(cellstr(textStrings));
[xGrid, yGrid] = meshgrid(1:2, 1:2);
hStrings = text(xGrid(:), yGrid(:), textStrings(:), ...
    'HorizontalAlignment', 'center', ...
    'Color', 'red', 'FontSize', 14, 'FontWeight', 'bold');

% Format axes ticks to display binary class labels ('0' and '1')
set(gca, 'XTick', 1:2, 'XTickLabel', {'0', '1'}, ...
    'YTick', 1:2, 'YTickLabel', {'0', '1'}, ...
    'TickLength', [0 0]);
xlabel('Predicted Class');
ylabel('True Class');
title('Test Set Confusion Matrix');