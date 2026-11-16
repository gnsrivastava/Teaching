%{ ========================================================================
% BACKGROUND & DATASET OVERVIEW: seqFISH Spatial Transcriptomics
% ========================================================================
% #Cell Type Prediction from seqFISH Data using Graph Convolutional Networks (GCN)
% This notebook demonstrates how to perform cell type classification on spatial transcriptomics data using a Graph Convolutional Network (GCN).
% The model leverages both gene expression and spatial neighborhood information to predict cell identities within tissue sections.
% 
% It includes:
% **Data Loading:** Uses the built-in seqFISH mouse cortex dataset
% **Graph Construction:** Builds a spatial neighbor graph based on cell proximity, connecting nearby cells in the tissue.
% **Model Training:** Trains a GCN for node (cell) classification using expression features and the spatial graph structure.
% **Performance Evaluation:** Computes accuracy and visualizes a normalized confusion matrix to assess classification performance.
%
%  #Dataset Overview
% 
% **Dataset:** seqFISH Mouse Cortex (from Squidpy)
% **Nodes (cells):** Each node represents an individual cell with its gene expression vector.
% **Edges:** Constructed based on spatial proximity between cells, representing their physical neighborhood relationships.
% **Labels:** Each cell is annotated with a refined reference cell type (e.g., neuron, astrocyte, endothelial cell).
%}

% Clear workspace variables, reset command window, and close all figure windows
clear; clc; close all

%% 1. Load and inspect the data
% Decompress the .mat.gz archive into the system temporary folder and load it
unpackedFile = gunzip('seqfish.mat.gz', tempdir);
S = load(unpackedFile{1});
delete(unpackedFile{1}); % Clean up temporary uncompressed .mat file

% Extract node feature matrix (cells x genes), 2D spatial coordinates, and cell type labels
X = single(S.X);
xy = double(S.xy);
cellType = string(S.cellType(:));

% Verify that features, spatial coordinates, and labels have matching cell counts
assert(size(X,1) == size(xy,1) && size(X,1) == numel(cellType), ...
    "Rows must refer to the same cells.");

% Convert string labels into categorical array and extract dataset dimensions
y = categorical(cellType);
cellNames = categories(y);
numNodes = size(X,1);       % Total number of cells (graph nodes)
numFeatures = size(X,2);    % Number of measured genes per cell (input features)
numClasses = numel(cellNames); % Number of distinct annotated cell types

% Display dataset summary statistics
fprintf("%d cells, %d genes, %d cell types\n", numNodes, numFeatures, numClasses);

%% 2. Visualize spatial distribution
% Plot the 2D physical layout of cells colored by their ground-truth cell type
figure("Color","w");
gscatter(xy(:,1), xy(:,2), y, [], [], 8);
set(gca, "YDir", "reverse"); % Invert Y-axis to match microscopy image orientation
axis equal tight
xlabel("Spatial coordinate 1");
ylabel("Spatial coordinate 2");
title("seqFISH cells by mapped cell type");
legend("Location", "eastoutside");

%% 3. Build normalized adjacency matrix
% Define neighborhood size for spatial k-Nearest Neighbors (k-NN) graph
neighborCount = 8;

% Find k+1 nearest neighbors for each cell (column 1 is the cell itself, so exclude it)
idx = knnsearch(xy, xy, "K", neighborCount + 1);
src = repelem((1:numNodes)', neighborCount); % Source node indices
dst = reshape(idx(:,2:end), [], 1);          % Target neighbor node indices

% Construct sparse adjacency matrix and symmetrize it to form an undirected graph
A = sparse(src, dst, 1, numNodes, numNodes);
A = spones(A + A');

% Add self-loops (A_tilde = A + I) so each cell includes its own gene expression during aggregation
A = A + speye(numNodes);

% Compute symmetric normalized adjacency matrix: A_hat = D^(-1/2) * (A + I) * D^(-1/2)
D = sum(A, 2); % Node degree vector
Ahat = spdiags(1./sqrt(D), 0, numNodes, numNodes) * A * spdiags(1./sqrt(D), 0, numNodes, numNodes);

% Visualize spatial graph connectivity on a subset of the first 500 cells
figure("Color","w");
nPlot = min(500, numNodes);
gplot(A(1:nPlot, 1:nPlot), xy(1:nPlot,:), "-"); % Plot spatial edges
hold on;
scatter(xy(:,1), xy(:,2), 5, double(y), "filled"); % Overlay all cell positions colored by type
axis equal tight
subtitle("Graph edges among first 500 cells");

%% 4. Stratified train/test split
% Set random seed for reproducible node splitting (transductive learning setup)
rng(7);
trainMask = false(numNodes, 1);
testMask = false(numNodes, 1);

% Perform an 80/20 stratified split across each cell type to preserve class proportions
for k = 1:numClasses
    members = find(y == cellNames{k});              % Indices of cells belonging to class k
    members = members(randperm(numel(members)));    % Randomly shuffle class members
    nTest = max(1, round(0.2 * numel(members)));    % Allocate 20% (at least 1 cell) to test set
    testMask(members(1:nTest)) = true;              % Assign test node mask
    trainMask(members(nTest+1:end)) = true;         % Assign remaining 80% to training mask
end

%% 5. Define GCN architecture using dlnetwork
hiddenSize = 64; % Dimensionality of hidden node embeddings

% Assemble a 2-layer Graph Convolutional Network operating on the full spatial graph
layers = [
    featureInputLayer(numFeatures, "Name", "input", "Normalization", "none")
    graphConvLayer(numFeatures, hiddenSize, Ahat, "gcn1") % Layer 1: Spatial neighborhood aggregation
    reluLayer("Name", "relu1")                            % Non-linear activation
    dropoutLayer(0.3, "Name", "drop1")                    % Regularization to prevent overfitting
    graphConvLayer(hiddenSize, numClasses, Ahat, "gcn2")  % Layer 2: Class logit projection
    softmaxLayer("Name", "softmax")                       % Convert logits to class probabilities
];

% Initialize deep learning network object
net = dlnetwork(layers);

%% 6. Train the GCN network
% Training hyperparameters
epochs = 200;           % Total full-graph training iterations
learnRate = 0.01;       % Adam optimizer step size
weightDecay = 5e-4;     % L2 regularization penalty coefficient

% Initialize Adam optimizer state variables and metric tracking arrays
trailingAvg = [];
trailingAvgSq = [];
lossHistory = zeros(epochs, 1);
accuracyHistory = zeros(epochs, 1);

% Format feature matrix as dlarray with Channel ('C', genes) x Batch ('B', cells) layout
Xdl = dlarray(X', "CB");

% Full-batch transductive training loop
for epoch = 1:epochs
    % Compute masked cross-entropy loss and parameter gradients via automatic differentiation
    [loss, gradients] = dlfeval(@modelLoss, net, Xdl, y, trainMask, weightDecay);
    
    % Update learnable network weights using Adam
    [net, trailingAvg, trailingAvgSq] = adamupdate( ...
        net, gradients, trailingAvg, trailingAvgSq, epoch, learnRate);
        
    % Record scalar training loss for current epoch
    lossHistory(epoch) = double(extractdata(loss));
    
    % Evaluate model predictions on held-out test nodes
    scores = predict(net, Xdl);
    [~, predIdx] = max(extractdata(scores), [], 1);
    pred = categorical(predIdx', 1:numClasses, cellNames);
    accuracyHistory(epoch) = mean(pred(testMask) == y(testMask));
    
    % Log progress every 20 epochs
    if mod(epoch, 20) == 0
        fprintf("Epoch %3d | Loss: %.4f | Test Accuracy: %.3f\n", ...
            epoch, lossHistory(epoch), accuracyHistory(epoch));
    end
end

%% 7. Evaluate and plot results
% Generate final class predictions across all cells and compute test set accuracy
scores = predict(net, Xdl);
[~, predIdx] = max(extractdata(scores), [], 1);
pred = categorical(predIdx', 1:numClasses, cellNames);
acc = mean(pred(testMask) == y(testMask));
fprintf("Final Test Accuracy: %.3f\n", acc);

% Plot training loss and test accuracy learning curves side-by-side
figure("Color","w");
tiledlayout(1,2);

nexttile;
plot(lossHistory, "LineWidth", 1.5);
xlabel("Epoch"); ylabel("Loss"); title("Training Loss"); grid on;

nexttile;
plot(accuracyHistory, "LineWidth", 1.5);
xlabel("Epoch"); ylabel("Accuracy"); title("Test Accuracy");
ylim([0 1]); grid on;

% Compute confusion matrix on test cells and normalize each row to percentages
cm = confusionmat(y(testMask), pred(testMask), "Order", categorical(cellNames));
cm = 100 * cm ./ max(sum(cm, 2), 1);

% Display row-normalized confusion matrix heatmap across all cell types
figure("Color","w");
heatmap(cellNames, cellNames, cm, "ColorLimits", [0 100]);
xlabel("Predicted Cell Type");
ylabel("True Cell Type");
title("Row-Normalized Confusion Matrix (%)");

%% Helper Functions
function [loss, gradients] = modelLoss(net, X, y, mask, weightDecay)
    % Perform forward pass through the entire spatial graph
    scores = forward(net, X);
    
    % Filter predictions to include only training nodes specified by 'mask'
    scoresMasked = scores(:, mask);
    
    % Convert categorical labels for masked nodes into a one-hot target matrix [Classes x Nodes]
    labelsMasked = y(mask);
    numClasses = size(scores, 1);
    numMasked = numel(labelsMasked);
    T = zeros(numClasses, numMasked, "single");
    T(sub2ind([numClasses, numMasked], double(labelsMasked)', 1:numMasked)) = 1;
    targetsMasked = dlarray(T, "CB");
    
    % Compute multi-class cross-entropy loss on training nodes
    loss = crossentropy(scoresMasked, targetsMasked);
    
    % Add L2 weight decay regularization penalty across all learnable parameters
    for i = 1:height(net.Learnables)
        loss = loss + weightDecay * sum(net.Learnables.Value{i}.^2, "all");
    end
    
    % Compute gradients of total loss with respect to network parameters
    gradients = dlgradient(loss, net.Learnables);
end