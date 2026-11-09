%% 1. Environment Setup & Configuration
clear; clc; rng(42);

matFileName = 'BBBP.mat'; 

%% 2. Load Dataset
if ~isfile(matFileName)
    error('File "%s" not found. Please ensure the .mat file is in the current working directory.', matFileName);
end

fprintf('Loading dataset from %s...\n', matFileName);
dataStruct = load(matFileName);

if isfield(dataStruct, 'X') && isfield(dataStruct, 'labels')
    X = single(dataStruct.X);          % Feature matrix [N x NumFeatures]
    labels = double(dataStruct.labels); % Binary targets [N x 1]
else
    error('The specified .mat file must contain variables named "X" and "labels".');
end

[numSamples, numFeatures] = size(X);

%% 3. Prepare Batch Data for dlnetwork
batchSize = min(64, numSamples);
xBatch = dlarray(X(1:batchSize, :)', 'CB'); % [Channels x Batch]

% Transpose categorical labels to [1 x Batch] so dimension 1 is singleton
yBatchCat = categorical(labels(1:batchSize))'; 
yBatch = onehotencode(yBatchCat, 1);        % Output shape: [2 x Batch]
yBatch = dlarray(yBatch, 'CB');

%% 4. Define Network Architecture as dlnetwork
layers = [
    featureInputLayer(numFeatures, 'Name', 'input')
    fullyConnectedLayer(32, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(2, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    ];

net = dlnetwork(layers);

%% 5. Compute Gradients via Automatic Differentiation
fprintf('Computing backpropagation gradients...\n');
[loss, gradients] = dlfeval(@modelLoss, net, xBatch, yBatch);

%% 6. Extract and Visualize Gradient Distribution
fc1WeightIdx = strcmp(gradients.Parameter, 'Weights') & strcmp(gradients.Layer, 'fc1');
gradFC1 = double(extractdata(gradients.Value{fc1WeightIdx}));

figure('Name', 'Backpropagation Gradient Distribution', 'Position', [100, 100, 600, 400]);
histogram(gradFC1(:), 50, 'FaceColor', '#0072BD', 'EdgeColor', 'none');
xlabel('Gradient Value (\partial L / \partial W)');
ylabel('Frequency');
title(sprintf('Gradient Distribution across FC1 Weights (Loss: %.4f)', double(extractdata(loss))));
grid on;

fprintf('Gradient visualization complete.\n');

%% =========================================================================
%% LOCAL HELPER FUNCTIONS
%% =========================================================================
function [loss, gradients] = modelLoss(net, X, Y)
YPred = forward(net, X);
loss = crossentropy(YPred, Y);
gradients = dlgradient(loss, net.Learnables);
end