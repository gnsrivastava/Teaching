%% ========================================================================
%  BACKGROUND & DATASET OVERVIEW: BACE1 Molecular Graph Classification
%  ========================================================================
%  Biological Target:
%    Human beta-secretase 1 (BACE1) is a critical transmembrane aspartyl 
%    protease involved in the cleavage of amyloid precursor protein (APP). 
%    Inhibition of BACE1 reduces the formation of amyloid-beta (Abeta) 
%    peptides, making it a major therapeutic target for Alzheimer's disease.
%
%  Dataset Description:
%    - Source: MoleculeNet / DeepChem Benchmark Suite.
%    - Total Compounds: ~1,513 small molecules with experimentally measured 
%      binding affinities (IC50 / pIC50).
%    - Task Type: Binary Classification.
%        * Active (1)   : High binding affinity (low IC50 / pIC50 >= 6.0).
%        * Inactive (0) : Low/no inhibition (IC50 > 1 uM / pIC50 < 6.0).
%    - Graph Representation:
%        * Nodes (V)    : Atoms, featurized with atomic number, hybridization, 
%                         aromaticity, formal charge, and degree.
%        * Edges (E)    : Chemical bonds represented in the adjacency matrix.
%
%  Evaluation Challenge (Bemis-Murcko Scaffold Split):
%    - To simulate real-world drug discovery, molecules are split based on 
%      their core chemical scaffolds (~80% train, 10% validation, 10% test).
%    - Test molecules contain distinct core backbones not present in training, 
%      testing the model's ability to generalize to novel chemical space.
% ========================================================================

% Clear workspace variables, reset command window, and close figure windows
clear;
clc;
close all;

%% 1. Load Preprocessed Graph Data
matFile = "bace_graphs.mat"; % Specify the path to the MAT-file containing dataset graphs

% Load the 'graphs' array and external 'drugbank' evaluation set from the file
loaded = load(matFile, "graphs", "drugbank");

% Unpack variables from the loaded structure into local variables
graphs   = loaded.graphs;   % Main graph dataset containing X (features), A (adjacency), y (labels)
drugbank = loaded.drugbank; % External set of compounds for screening inference

% Print summary statistics: total graph count and fraction of active (y=1) compounds
fprintf("Loaded %d graphs; active fraction %.3f.\n", ...
    numel(graphs), mean(double([graphs.y])));

%% 2. Bemis-Murcko Scaffold Split (Identical Seed = 7)
% Split dataset into train/val/test using scaffold grouping and random seed 7 for reproducibility
[trainIdx, valIdx, testIdx] = scaffoldSplit({graphs.scaffold}, 7);

% Print the number of molecules allocated to each split
fprintf("Scaffold split: train %d, validation %d, test %d.\n", ...
    numel(trainIdx), numel(valIdx), numel(testIdx));

% Subset the main graph array into separate arrays using the split indices
trainGraphs = graphs(trainIdx); % Training set (~80%)
valGraphs   = graphs(valIdx);   % Validation set (~10%)
testGraphs  = graphs(testIdx);  % Test set (~10%)

%% 3. Model Hyperparameters
numFeatures = size(graphs(1).X, 2); % Number of input features per atom (node feature dimension)
numClasses  = 2;                    % Binary classification: 0 (Inactive), 1 (Active)
hiddenUnits = 32;                   % Dimensionality of hidden node representation vectors
maxEpochs   = 100;                  % Maximum number of full passes over the training dataset
batchSize   = 16;                   % Minibatch size (number of graphs processed per iteration)
lr          = 0.003;                % Learning rate for the Adam optimizer

%% 4. Train Standard Graph Convolutional Network (GCN)
fprintf("\n==================================================\n");
fprintf("              Training GCN Model                  \n");
fprintf("==================================================\n");

% Set random number generator seed for weight initialization reproducibility
rng(42, "twister");

% Initialize GCN model parameters as dlarrays inside a structure
paramsGCN = struct();
% Layer 1 weight matrix: maps input features (numFeatures) to hidden dim (hiddenUnits)
paramsGCN.gcn1_W = initializeGlorot([numFeatures, hiddenUnits]);
% Layer 2 weight matrix: maps hidden features to hidden features
paramsGCN.gcn2_W = initializeGlorot([hiddenUnits, hiddenUnits]);
% Fully connected classification layer weight matrix
paramsGCN.fc_W   = initializeGlorot([hiddenUnits, numClasses]);
% Fully connected classification layer bias vector
paramsGCN.fc_b   = dlarray(zeros(1, numClasses, "single"));

% Initialize moving average variables for Adam optimizer
avgGradGCN = []; 
avgSqGradGCN = []; 
iterGCN = 0; % Track global optimization step counter

% Main training loop over epochs
for epoch = 1:maxEpochs
    % Shuffle training graphs randomly at the beginning of each epoch
    order = randperm(numel(trainGraphs));
    totalLoss = 0; % Accumulate total epoch loss

    % Minibatch loop
    for first = 1:batchSize:numel(order)
        % Determine indices of current minibatch
        last = min(first + batchSize - 1, numel(order));
        batchIdx = order(first:last);

        % Combine minibatch graphs into block-diagonal matrices and symmetric normalized adjacency
        [A_norm, X, Y, graphIdx] = prepareBatchGCN(trainGraphs(batchIdx));

        % Evaluate model loss and parameter gradients using dlfeval (enables automatic differentiation)
        [loss, grads] = dlfeval(@modelLossGCN, paramsGCN, X, A_norm, graphIdx, Y);

        % Increment Adam global step counter
        iterGCN = iterGCN + 1;

        % Update model parameters using the Adam optimization algorithm
        [paramsGCN, avgGradGCN, avgSqGradGCN] = adamupdate( ...
            paramsGCN, grads, avgGradGCN, avgSqGradGCN, iterGCN, lr);

        % Add minibatch loss weighted by batch size to total epoch loss
        totalLoss = totalLoss + double(extractdata(loss)) * numel(batchIdx);
    end

    % Evaluate and display validation loss every 20 epochs and on epoch 1
    if mod(epoch, 20) == 0 || epoch == 1
        valLoss = evaluateLossGCN(paramsGCN, valGraphs, batchSize);
        fprintf("Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f\n", ...
            epoch, maxEpochs, totalLoss/numel(trainGraphs), valLoss);
    end
end

%% 5. Train Enhanced Multi-Head GAT Model
fprintf("\n==================================================\n");
fprintf("       Training Multi-Head Residual GAT Model     \n");
fprintf("==================================================\n");

% Reset random seed so GAT initialization starts from comparable conditions
rng(42, "twister");

% Multi-Head Attention Configuration
numHeads = 4;  % Use 4 parallel attention heads
headDim  = 16; % Dimension per head (4 heads * 16 dims = 64 intermediate channels)

paramsGAT = struct();
% Layer 1 Multi-Head linear transformation weights: maps input features to total head output dims
paramsGAT.gat1_W    = initializeGlorot([numFeatures, headDim * numHeads]);
% Layer 1 Multi-Head attention weight vectors (a1 and a2 concatenated per head)
paramsGAT.gat1_a    = initializeGlorot([2 * headDim, numHeads]);
% Layer 1 Residual (skip connection) linear projection matrix
paramsGAT.gat1_Wres = initializeGlorot([numFeatures, headDim * numHeads]);

% Layer 2 Single-Head aggregation transformation weights: maps 64 head dims to 32 hidden units
paramsGAT.gat2_W    = initializeGlorot([headDim * numHeads, hiddenUnits]);
% Layer 2 Single-Head attention weight vector
paramsGAT.gat2_a    = initializeGlorot([2 * hiddenUnits, 1]);
% Layer 2 Residual (skip connection) linear projection matrix
paramsGAT.gat2_Wres = initializeGlorot([headDim * numHeads, hiddenUnits]);

% Final classification layer weights and bias
paramsGAT.fc_W   = initializeGlorot([hiddenUnits, numClasses]);
paramsGAT.fc_b   = dlarray(zeros(1, numClasses, "single"));

% Initialize Adam momentum variables for GAT
avgGradGAT = []; 
avgSqGradGAT = []; 
iterGAT = 0;

% Main training loop over epochs for GAT
for epoch = 1:maxEpochs
    % Randomly permute training graph order
    order = randperm(numel(trainGraphs));
    totalLoss = 0;

    % Minibatch loop
    for first = 1:batchSize:numel(order)
        last = min(first + batchSize - 1, numel(order));
        batchIdx = order(first:last);

        % Assemble minibatch graph data into block-diagonal format for GAT
        [A, X, Y, graphIdx] = prepareBatchGAT(trainGraphs(batchIdx));

        % Compute GAT loss and gradients using automatic differentiation
        [loss, grads] = dlfeval(@modelLossGAT, paramsGAT, X, A, graphIdx, Y, numHeads);

        % Update step iteration counter
        iterGAT = iterGAT + 1;

        % Apply Adam parameter updates
        [paramsGAT, avgGradGAT, avgSqGradGAT] = adamupdate( ...
            paramsGAT, grads, avgGradGAT, avgSqGradGAT, iterGAT, lr);

        % Accumulate epoch loss
        totalLoss = totalLoss + double(extractdata(loss)) * numel(batchIdx);
    end

    % Periodically evaluate validation loss
    if mod(epoch, 20) == 0 || epoch == 1
        valLoss = evaluateLossGAT(paramsGAT, valGraphs, batchSize, numHeads);
        fprintf("Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f\n", ...
            epoch, maxEpochs, totalLoss/numel(trainGraphs), valLoss);
    end
end

%% 6. Comparative Evaluation & Print Summary Table
% Define anonymous function handles for prediction to standardize metric computation interface
predictGCN_fn = @(p, g, b) predictGCN(p, g, b);
predictGAT_fn = @(p, g, b) predictGAT(p, g, b, numHeads);

% Calculate confusion matrix, accuracy, sensitivity, specificity, and ROC-AUC for both models
metricsGCN = computeMetrics(paramsGCN, testGraphs, batchSize, predictGCN_fn);
metricsGAT = computeMetrics(paramsGAT, testGraphs, batchSize, predictGAT_fn);

% Print formatted comparative evaluation table to command window
fprintf("\n========================================================\n");
fprintf("       Model Performance Summary (Test Set n=%d)        \n", numel(testGraphs));
fprintf("========================================================\n");
fprintf(" Metric                      |   GCN Model  | Multi-Head GAT\n");
fprintf("-----------------------------+--------------+-----------\n");
fprintf(" Accuracy                    |    %5.1f%%    |    %5.1f%%   \n", metricsGCN.acc*100, metricsGAT.acc*100);
fprintf(" ROC-AUC                     |    %6.3f     |    %6.3f    \n", metricsGCN.auc, metricsGAT.auc);
fprintf(" Class 0 Specificity         |    %5.1f%%    |    %5.1f%%   \n", metricsGCN.spec*100, metricsGAT.spec*100);
fprintf(" Class 0 NPV (Precision)     |    %5.1f%%    |    %5.1f%%   \n", metricsGCN.npv*100, metricsGAT.npv*100);
fprintf(" Class 1 Sensitivity         |    %5.1f%%    |    %5.1f%%   \n", metricsGCN.sens*100, metricsGAT.sens*100);
fprintf(" Class 1 PPV (Precision)     |    %5.1f%%    |    %5.1f%%   \n", metricsGCN.ppv*100, metricsGAT.ppv*100);
fprintf("-----------------------------+--------------+-----------\n");
fprintf(" Confusion Matrix (TN/FP/FN/TP)| %2d/%2d/%2d/%2d | %2d/%2d/%2d/%2d\n", ...
    metricsGCN.TN, metricsGCN.FP, metricsGCN.FN, metricsGCN.TP, ...
    metricsGAT.TN, metricsGAT.FP, metricsGAT.FN, metricsGAT.TP);
fprintf("========================================================\n");

%% 7. Inference on External DrugBank Molecules
% Perform comparative inference on external molecules (known controls and BACE inhibitors)
if ~isempty(drugbank)
    % Compute raw model scores for DrugBank molecules
    scoresDB_GCN = predictGCN(paramsGCN, drugbank, batchSize);
    probsDB_GCN  = extractdata(softmax(scoresDB_GCN, "DataFormat", "CB"));

    scoresDB_GAT = predictGAT(paramsGAT, drugbank, batchSize, numHeads);
    probsDB_GAT  = extractdata(softmax(scoresDB_GAT, "DataFormat", "CB"));

    % Construct comparative table with predicted probabilities of active class (row 2)
    externalComp = table( ...
        reshape(string({drugbank.DrugBankID}), [], 1), ...
        reshape(string({drugbank.Name}), [], 1), ...
        probsDB_GCN(2, :)', ...
        probsDB_GAT(2, :)', ...
        'VariableNames', {'DrugBankID', 'Name', 'GCN_Prob_Active', 'GAT_Prob_Active'});

    fprintf("\n--- External DrugBank Predictions ---\n");
    disp(externalComp);
end

%% --- Local Functions: Enhanced Multi-Head GAT ---

function [loss, grads] = modelLossGAT(params, X, A, graphIdx, Y, numHeads)
    % Compute model output classification scores
    scores = forwardGAT(params, X, A, graphIdx, numHeads);
    
    % Apply softmax function along channel dimension (C=2 classes)
    probs  = softmax(scores, "DataFormat", "CB");
    
    % Define class weight penalty vector (1.0 for Inactive, 1.5 for Active to penalize false negatives)
    weights = dlarray([1.0; 1.5], "CB");
    
    % Apply element-wise class weighting to predicted probabilities
    weightedProbs = probs .* weights;
    
    % Compute cross-entropy loss without repeating DataFormat (since weightedProbs is formatted)
    loss  = crossentropy(weightedProbs, Y);
    
    % Automatically compute loss gradients with respect to learnable model parameters
    grads = dlgradient(loss, params);
end

function scores = forwardGAT(params, X, A, graphIdx, numHeads)
    % Layer 1: Compute multi-head attention outputs (concatenated across heads)
    H1_att = gatMultiHeadCompute(X, A, params.gat1_W, params.gat1_a, numHeads);
    % Layer 1: Linear residual projection to match feature dimensions
    H1_res = X * params.gat1_Wres;
    % Apply ELU activation function to combined attention and residual vectors
    H1     = eluActivation(H1_att + H1_res);

    % Layer 2: Compute single-head output aggregation attention
    H2_att = gatCompute(H1, A, params.gat2_W, params.gat2_a);
    % Layer 2: Linear residual projection
    H2_res = H1 * params.gat2_Wres;
    % Apply ELU activation to second layer output
    H2     = eluActivation(H2_att + H2_res);

    % Aggregate node representations per graph via mean pooling
    H_pool = graphMeanPooling(H2, graphIdx);
    
    % Linear output projection to compute class scores (logits)
    scores = params.fc_W' * H_pool + params.fc_b';
end

function H_out = gatMultiHeadCompute(X, A, W, a, numHeads)
    % Linear feature projection for all attention heads simultaneously
    Z_all = X * W;
    
    % Calculate output feature dimension per head
    headDim = size(Z_all, 2) / numHeads;
    headOutputs = cell(1, numHeads);

    % Iterate through each attention head independently
    for h = 1:numHeads
        % Extract slice of projected features belonging to current head h
        idxRange = (h-1)*headDim + (1:headDim);
        Z = Z_all(:, idxRange);

        % Compute attention vector projections (source and target node terms)
        Z1 = Z * a(1:headDim, h);
        Z2 = Z * a(headDim+1:end, h);

        % Compute unnormalized attention coefficients using LeakyReLU activation
        E = leakyrelu(Z1 + Z2', 0.2);
        
        % Create mask to set non-neighboring graph edges (A == 0) to a large negative scalar
        nonNeighborMask = single(A == 0);
        E_masked = E + (-1e9) .* nonNeighborMask;

        % Apply Softmax over each row's neighbors to get normalized attention matrix Alpha
        Alpha = softmax(E_masked', "DataFormat", "CB")';
        
        % Aggregate neighbor features weighted by attention coefficients
        headOutputs{h} = Alpha * Z;
    end
    
    % Concatenate features from all attention heads along feature dimension
    H_out = cat(2, headOutputs{:});
end

function H_out = gatCompute(X, A, W, a)
    % Single-head attention compute function used in Layer 2
    Z = X * W;
    numOutputs = size(Z, 2);
    
    % Split single-head attention vector into source (a1) and target (a2) halves
    Z1 = Z * a(1:numOutputs);
    Z2 = Z * a(numOutputs + 1:end);

    % Compute pairwise attention logits
    E = leakyrelu(Z1 + Z2', 0.2);
    
    % Mask non-connected nodes
    nonNeighborMask = single(A == 0);
    E_masked = E + (-1e9) .* nonNeighborMask;

    % Normalize attention coefficients via Softmax
    Alpha = softmax(E_masked', "DataFormat", "CB")';
    
    % Compute weighted node feature aggregation
    H_out = Alpha * Z;
end

function scores = predictGAT(params, graphStructs, batchSize, numHeads)
    % Perform batched forward inference on a set of graph structures
    numGraphs = numel(graphStructs);
    numClasses = size(params.fc_W, 2);
    scoresAll = zeros(numClasses, numGraphs, "single");
    
    % Minibatch evaluation loop
    for first = 1:batchSize:numGraphs
        last = min(first + batchSize - 1, numGraphs);
        idx = first:last;
        [A, X, ~, graphIdx] = prepareBatchGAT(graphStructs(idx));
        % Evaluate forward pass and extract raw score values from dlarray
        scoresAll(:, idx) = extractdata(forwardGAT(params, X, A, graphIdx, numHeads));
    end
    % Wrap result into formatted dlarray
    scores = dlarray(scoresAll);
end

function lossVal = evaluateLossGAT(params, graphStructs, batchSize, numHeads)
    % Compute validation/evaluation cross-entropy loss without gradient tracking
    numGraphs = numel(graphStructs);
    totalLoss = 0;
    for first = 1:batchSize:numGraphs
        last = min(first + batchSize - 1, numGraphs);
        idx = first:last;
        [A, X, Y, graphIdx] = prepareBatchGAT(graphStructs(idx));
        scores = forwardGAT(params, X, A, graphIdx, numHeads);
        probs  = softmax(scores, "DataFormat", "CB");
        loss   = crossentropy(probs, Y, "DataFormat", "CB");
        totalLoss = totalLoss + double(extractdata(loss)) * numel(idx);
    end
    lossVal = totalLoss / numGraphs;
end

function [A_block, X_cat, Y_onehot, graphIdx] = prepareBatchGAT(graphStructs)
    % Prepare block-diagonal adjacency and concatenated node features for GAT minibatch
    numGraphs = numel(graphStructs);
    numNodesTotal = sum(arrayfun(@(g) size(g.X, 1), graphStructs));
    numFeatures = size(graphStructs(1).X, 2);

    % Allocate output batch matrices
    A_block = zeros(numNodesTotal, numNodesTotal, "single");
    X_cat = zeros(numNodesTotal, numFeatures, "single");
    graphIdx = zeros(numNodesTotal, 1);
    hasLabels = isfield(graphStructs, "y");
    Y_onehot = zeros(2, numGraphs, "single");

    currIdx = 0;
    % Loop over each graph in minibatch to build block-diagonal representations
    for i = 1:numGraphs
        n = size(graphStructs(i).X, 1);
        idxRange = currIdx + (1:n);

        % Extract adjacency matrix and add self-loops (diagonal = 1)
        adjacency = single(full(graphStructs(i).A) ~= 0);
        adjacency(1:n + 1:end) = 1;

        % Place adjacency into non-overlapping diagonal block
        A_block(idxRange, idxRange) = adjacency;
        X_cat(idxRange, :) = single(full(graphStructs(i).X));
        graphIdx(idxRange) = i; % Map each node to its graph ID in the batch

        % Format binary label (0 or 1) as a one-hot vector column
        if hasLabels
            Y_onehot(double(graphStructs(i).y) + 1, i) = 1;
        end
        currIdx = currIdx + n;
    end
    % Convert to dlarray types
    X_cat    = dlarray(X_cat);
    Y_onehot = dlarray(Y_onehot);
end

%% --- Local Functions: Standard GCN ---

function [loss, grads] = modelLossGCN(params, X, A_norm, graphIdx, Y)
    % Forward pass to get output scores
    scores = forwardGCN(params, X, A_norm, graphIdx);
    % Convert scores to probabilities via softmax
    probs  = softmax(scores, "DataFormat", "CB");
    % Compute standard cross-entropy loss
    loss   = crossentropy(probs, Y, "DataFormat", "CB");
    % Compute automatic differentiation gradients
    grads  = dlgradient(loss, params);
end

function scores = forwardGCN(params, X, A_norm, graphIdx)
    % GCN Layer 1: H1 = ELU( A_norm * X * W1 )
    H1 = eluActivation(A_norm * (X * params.gcn1_W));
    % GCN Layer 2: H2 = ELU( A_norm * H1 * W2 )
    H2 = eluActivation(A_norm * (H1 * params.gcn2_W));
    % Global mean pooling per graph
    H_pool = graphMeanPooling(H2, graphIdx);
    % Output linear layer
    scores = params.fc_W' * H_pool + params.fc_b';
end

function scores = predictGCN(params, graphStructs, batchSize)
    % Batched inference pipeline for GCN
    numGraphs = numel(graphStructs);
    numClasses = size(params.fc_W, 2);
    scoresAll = zeros(numClasses, numGraphs, "single");
    for first = 1:batchSize:numGraphs
        last = min(first + batchSize - 1, numGraphs);
        idx = first:last;
        [A_norm, X, ~, graphIdx] = prepareBatchGCN(graphStructs(idx));
        scoresAll(:, idx) = extractdata(forwardGCN(params, X, A_norm, graphIdx));
    end
    scores = dlarray(scoresAll);
end

function lossVal = evaluateLossGCN(params, graphStructs, batchSize)
    % Evaluation loss computation for GCN
    numGraphs = numel(graphStructs);
    totalLoss = 0;
    for first = 1:batchSize:numGraphs
        last = min(first + batchSize - 1, numGraphs);
        idx = first:last;
        [A_norm, X, Y, graphIdx] = prepareBatchGCN(graphStructs(idx));
        scores = forwardGCN(params, X, A_norm, graphIdx);
        probs = softmax(scores, "DataFormat", "CB");
        loss = crossentropy(probs, Y, "DataFormat", "CB");
        totalLoss = totalLoss + double(extractdata(loss)) * numel(idx);
    end
    lossVal = totalLoss / numGraphs;
end

function [A_norm, X_cat, Y_onehot, graphIdx] = prepareBatchGCN(graphStructs)
    % Assemble batch and compute symmetric normalized adjacency matrix for GCN
    numGraphs = numel(graphStructs);
    numNodesTotal = sum(arrayfun(@(g) size(g.X, 1), graphStructs));
    numFeatures = size(graphStructs(1).X, 2);

    A_block = zeros(numNodesTotal, numNodesTotal, "single");
    X_cat = zeros(numNodesTotal, numFeatures, "single");
    graphIdx = zeros(numNodesTotal, 1);
    hasLabels = isfield(graphStructs, "y");
    Y_onehot = zeros(2, numGraphs, "single");

    currIdx = 0;
    for i = 1:numGraphs
        n = size(graphStructs(i).X, 1);
        idxRange = currIdx + (1:n);

        % Add self-loops to Adjacency matrix: A_tilde = A + I
        A_tilde = single(full(graphStructs(i).A) ~= 0);
        A_tilde(1:n+1:end) = 1;

        % Compute degree vector D_tilde and inverse square root D_tilde^(-1/2)
        deg = sum(A_tilde, 2);
        degInvSqrt = deg .^ (-0.5);
        degInvSqrt(isinf(degInvSqrt) | isnan(degInvSqrt)) = 0;

        % Compute symmetric normalized sub-block: D^(-1/2) * A_tilde * D^(-1/2)
        A_block(idxRange, idxRange) = diag(degInvSqrt) * A_tilde * diag(degInvSqrt);
        X_cat(idxRange, :) = single(full(graphStructs(i).X));
        graphIdx(idxRange) = i;

        if hasLabels
            Y_onehot(double(graphStructs(i).y) + 1, i) = 1;
        end
        currIdx = currIdx + n;
    end
    % Wrap normalized adjacency and node features as dlarrays
    A_norm   = dlarray(A_block);
    X_cat    = dlarray(X_cat);
    Y_onehot = dlarray(Y_onehot);
end

%% --- Utility & Metrics Functions ---

function m = computeMetrics(params, testGraphs, batchSize, predictFn)
    % Evaluate predictions and calculate performance metrics (Accuracy, ROC-AUC, Sensitivity, Specificity)
    scoresTest = predictFn(params, testGraphs, batchSize);
    probsTest  = extractdata(softmax(scoresTest, "DataFormat", "CB"));

    % Convert probability scores to discrete class predictions (0 or 1)
    [~, yPredIdx] = max(probsTest, [], 1);
    yPred = reshape(yPredIdx - 1, [], 1);
    yTrue = reshape(double([testGraphs.y]), [], 1);

    % Compute 2x2 confusion matrix counts
    C = confusionmat(yTrue, yPred);
    m.TN = C(1,1); m.FP = C(1,2);
    m.FN = C(2,1); m.TP = C(2,2);

    % Calculate classification metrics
    m.acc  = mean(yTrue == yPred);                    % Overall Accuracy
    m.auc  = binaryAUC(yTrue, probsTest(2, :)');      % ROC-AUC for Class 1 (Active)
    m.spec = m.TN / (m.TN + m.FP);                     % Class 0 Specificity (True Negative Rate)
    m.sens = m.TP / (m.TP + m.FN);                     % Class 1 Sensitivity (True Positive Rate / Recall)
    m.npv  = m.TN / (m.TN + m.FN);                     % Class 0 Negative Predictive Value (Precision 0)
    m.ppv  = m.TP / (m.TP + m.FP);                     % Class 1 Positive Predictive Value (Precision 1)
end

function H_pool = graphMeanPooling(H, graphIdx)
    % Aggregate node representation vectors for each graph by averaging node features
    numGraphs = max(graphIdx);
    pooled = cell(1, numGraphs);
    for g = 1:numGraphs
        mask = (graphIdx == g);
        pooled{g} = mean(H(mask, :), 1)';
    end
    H_pool = cat(2, pooled{:});
end

function Y = eluActivation(X)
    % Exponential Linear Unit (ELU) activation function with alpha = 1.0
    Y = max(X, 0) + (exp(min(X, 0)) - 1);
end

function W = initializeGlorot(sz)
    % Initialize weight tensor parameters using Glorot (Xavier) uniform distribution
    bound = sqrt(6 / sum(sz));
    W = dlarray(single((2 * rand(sz) - 1) * bound));
end

function [tr, va, te] = scaffoldSplit(scaffolds, seed)
    % Initialize Mersenne Twister random number generator for split reproducibility
    rng(seed, "twister"); 
    
    % Convert the input scaffolds cell array or string array into a single row vector of strings
    key = reshape(string(scaffolds), 1, []); 
    
    % Find all unique molecular scaffold structures while preserving order of first appearance
    u = unique(key, "stable"); 
    
    % Group graph indices that share the exact same chemical scaffold backbone
    groups = cell(numel(u), 1);
    for k = 1:numel(u)
        groups{k} = reshape(find(key == u(k)), 1, []);
    end
    
    % Randomly permute scaffold groups (molecules inside each group stay bound together)
    groups = groups(randperm(numel(groups)));
   
    % Allocate Scaffold Groups sequentially to Train (~80%), Validation (~10%), and Test (~10%)
    n = numel(key);
    tr = []; va = []; te = [];
    for k = 1:numel(groups)
        g = groups{k};
        if numel(tr) < 0.8 * n
            tr = [tr, g];
        elseif numel(va) < 0.1 * n
            va = [va, g];
        else
            te = [te, g];
        end
    end
end

function auc = binaryAUC(y, p)
    % Compute exact Area Under the ROC Curve (ROC-AUC) using Mann-Whitney U / Rank-Sum statistic
    y = double(y(:)); p = double(p(:));
    n1 = sum(y == 1); n0 = sum(y == 0);
    if n1 == 0 || n0 == 0, auc = NaN; return; end
    
    % Sort predicted probabilities and compute ranks (handling tied values via average rank)
    [ps, ord] = sort(p);
    ranks = zeros(size(p));
    i = 1;
    while i <= numel(ps)
        j = i;
        while j < numel(ps) && ps(j + 1) == ps(i)
            j = j + 1;
        end
        ranks(ord(i:j)) = mean(i:j);
        i = j + 1;
    end
    
    % Calculate ROC-AUC from rank sum of positive class elements
    auc = (sum(ranks(y == 1)) - n1 * (n1 + 1) / 2) / (n1 * n0);
end