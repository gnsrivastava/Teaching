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
paramsGCN.gcn1_W = initializeGlorot([numFeatures, hiddenUnits]);
paramsGCN.gcn2_W = initializeGlorot([hiddenUnits, hiddenUnits]);
paramsGCN.fc_W   = initializeGlorot([hiddenUnits, numClasses]);
paramsGCN.fc_b   = dlarray(zeros(1, numClasses, "single"));

% Initialize moving average variables for Adam optimizer
avgGradGCN = []; 
avgSqGradGCN = []; 
iterGCN = 0; % Track global optimization step counter

% Preallocate metric history vectors for plotting
trainLossGCN = zeros(maxEpochs, 1);
valLossGCN   = zeros(maxEpochs, 1);
trainAccGCN  = zeros(maxEpochs, 1);
valAccGCN    = zeros(maxEpochs, 1);

% Main training loop over epochs
for epoch = 1:maxEpochs
    % Shuffle training graphs randomly at the beginning of each epoch
    order = randperm(numel(trainGraphs));
    totalLoss = 0;    % Accumulate total epoch loss
    totalCorrect = 0; % Accumulate correct training predictions
    
    % Minibatch loop
    for first = 1:batchSize:numel(order)
        last = min(first + batchSize - 1, numel(order));
        batchIdx = order(first:last);
        
        % Combine minibatch graphs into block-diagonal matrices and symmetric normalized adjacency
        [A_norm, X, Y, graphIdx] = prepareBatchGCN(trainGraphs(batchIdx));
        
        % Evaluate model loss, gradients, and minibatch predictions using dlfeval
        [loss, grads, batchProbs] = dlfeval(@modelLossGCN, paramsGCN, X, A_norm, graphIdx, Y);
        
        % Increment Adam global step counter and update parameters
        iterGCN = iterGCN + 1;
        [paramsGCN, avgGradGCN, avgSqGradGCN] = adamupdate( ...
            paramsGCN, grads, avgGradGCN, avgSqGradGCN, iterGCN, lr);
        
        % Accumulate minibatch loss and accuracy
        totalLoss = totalLoss + double(extractdata(loss)) * numel(batchIdx);
        [~, predIdx] = max(extractdata(batchProbs), [], 1);
        [~, trueIdx] = max(extractdata(Y), [], 1);
        totalCorrect = totalCorrect + sum(predIdx == trueIdx);
    end
    
    % Record epoch training and validation metrics
    trainLossGCN(epoch) = totalLoss / numel(trainGraphs);
    trainAccGCN(epoch)  = totalCorrect / numel(trainGraphs);
    [valLossGCN(epoch), valAccGCN(epoch)] = evaluateMetricsGCN(paramsGCN, valGraphs, batchSize);
    
    % Display progress every 20 epochs and on epoch 1
    if mod(epoch, 20) == 0 || epoch == 1
        fprintf("Epoch %3d/%d | Train Loss: %.4f (Acc: %5.1f%%) | Val Loss: %.4f (Acc: %5.1f%%)\n", ...
            epoch, maxEpochs, trainLossGCN(epoch), trainAccGCN(epoch)*100, ...
            valLossGCN(epoch), valAccGCN(epoch)*100);
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
paramsGAT.gat1_W    = initializeGlorot([numFeatures, headDim * numHeads]);
paramsGAT.gat1_a    = initializeGlorot([2 * headDim, numHeads]);
paramsGAT.gat1_Wres = initializeGlorot([numFeatures, headDim * numHeads]);
paramsGAT.gat2_W    = initializeGlorot([headDim * numHeads, hiddenUnits]);
paramsGAT.gat2_a    = initializeGlorot([2 * hiddenUnits, 1]);
paramsGAT.gat2_Wres = initializeGlorot([headDim * numHeads, hiddenUnits]);
paramsGAT.fc_W      = initializeGlorot([hiddenUnits, numClasses]);
paramsGAT.fc_b      = dlarray(zeros(1, numClasses, "single"));

% Initialize Adam momentum variables for GAT
avgGradGAT = []; 
avgSqGradGAT = []; 
iterGAT = 0;

% Preallocate metric history vectors for plotting
trainLossGAT = zeros(maxEpochs, 1);
valLossGAT   = zeros(maxEpochs, 1);
trainAccGAT  = zeros(maxEpochs, 1);
valAccGAT    = zeros(maxEpochs, 1);

% Main training loop over epochs for GAT
for epoch = 1:maxEpochs
    order = randperm(numel(trainGraphs));
    totalLoss = 0;
    totalCorrect = 0;
    
    % Minibatch loop
    for first = 1:batchSize:numel(order)
        last = min(first + batchSize - 1, numel(order));
        batchIdx = order(first:last);
        
        % Assemble minibatch graph data into block-diagonal format for GAT
        [A, X, Y, graphIdx] = prepareBatchGAT(trainGraphs(batchIdx));
        
        % Compute GAT loss, gradients, and minibatch predictions using automatic differentiation
        [loss, grads, batchProbs] = dlfeval(@modelLossGAT, paramsGAT, X, A, graphIdx, Y, numHeads);
        
        % Update step iteration counter and apply Adam updates
        iterGAT = iterGAT + 1;
        [paramsGAT, avgGradGAT, avgSqGradGAT] = adamupdate( ...
            paramsGAT, grads, avgGradGAT, avgSqGradGAT, iterGAT, lr);
        
        % Accumulate epoch loss and accuracy
        totalLoss = totalLoss + double(extractdata(loss)) * numel(batchIdx);
        [~, predIdx] = max(extractdata(batchProbs), [], 1);
        [~, trueIdx] = max(extractdata(Y), [], 1);
        totalCorrect = totalCorrect + sum(predIdx == trueIdx);
    end
    
    % Record epoch training and validation metrics
    trainLossGAT(epoch) = totalLoss / numel(trainGraphs);
    trainAccGAT(epoch)  = totalCorrect / numel(trainGraphs);
    [valLossGAT(epoch), valAccGAT(epoch)] = evaluateMetricsGAT(paramsGAT, valGraphs, batchSize, numHeads);
    
    % Periodically evaluate and print validation metrics
    if mod(epoch, 20) == 0 || epoch == 1
        fprintf("Epoch %3d/%d | Train Loss: %.4f (Acc: %5.1f%%) | Val Loss: %.4f (Acc: %5.1f%%)\n", ...
            epoch, maxEpochs, trainLossGAT(epoch), trainAccGAT(epoch)*100, ...
            valLossGAT(epoch), valAccGAT(epoch)*100);
    end
end

%% 6. Comparative Evaluation & Print Summary Table
predictGCN_fn = @(p, g, b) predictGCN(p, g, b);
predictGAT_fn = @(p, g, b) predictGAT(p, g, b, numHeads);

metricsGCN = computeMetrics(paramsGCN, testGraphs, batchSize, predictGCN_fn);
metricsGAT = computeMetrics(paramsGAT, testGraphs, batchSize, predictGAT_fn);

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
if ~isempty(drugbank)
    scoresDB_GCN = predictGCN(paramsGCN, drugbank, batchSize);
    probsDB_GCN  = extractdata(softmax(scoresDB_GCN, "DataFormat", "CB"));
    scoresDB_GAT = predictGAT(paramsGAT, drugbank, batchSize, numHeads);
    probsDB_GAT  = extractdata(softmax(scoresDB_GAT, "DataFormat", "CB"));

    externalComp = table( ...
        reshape(string({drugbank.DrugBankID}), [], 1), ...
        reshape(string({drugbank.Name}), [], 1), ...
        probsDB_GCN(2, :)', ...
        probsDB_GAT(2, :)', ...
        'VariableNames', {'DrugBankID', 'Name', 'GCN_Prob_Active', 'GAT_Prob_Active'});
    fprintf("\n--- External DrugBank Predictions ---\n");
    disp(externalComp);
end

%% 8. Plot Training & Validation Curves (Loss and Accuracy)
epochsVec = 1:maxEpochs;

% Plot 1: Training & Validation Loss
figure('Name', 'Training and Validation Loss', 'Color', 'w');
plot(epochsVec, trainLossGCN, '--', 'Color', [0.00, 0.45, 0.74], 'LineWidth', 1.5, 'DisplayName', 'GCN Train Loss');
hold on;
plot(epochsVec, valLossGCN,   '-',  'Color', [0.00, 0.45, 0.74], 'LineWidth', 2.0, 'DisplayName', 'GCN Val Loss');
plot(epochsVec, trainLossGAT, '--', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 1.5, 'DisplayName', 'GAT Train Loss');
plot(epochsVec, valLossGAT,   '-',  'Color', [0.85, 0.33, 0.10], 'LineWidth', 2.0, 'DisplayName', 'GAT Val Loss');
hold off;
grid on;
xlabel('Epoch', 'FontWeight', 'bold');
ylabel('Cross-Entropy Loss', 'FontWeight', 'bold');
title('BACE1 Classification: Training vs. Validation Loss', 'FontSize', 12);
legend('Location', 'best');

% Plot 2: Training & Validation Accuracy
figure('Name', 'Training and Validation Accuracy', 'Color', 'w');
plot(epochsVec, trainAccGCN * 100, '--', 'Color', [0.00, 0.45, 0.74], 'LineWidth', 1.5, 'DisplayName', 'GCN Train Accuracy');
hold on;
plot(epochsVec, valAccGCN   * 100, '-',  'Color', [0.00, 0.45, 0.74], 'LineWidth', 2.0, 'DisplayName', 'GCN Val Accuracy');
plot(epochsVec, trainAccGAT * 100, '--', 'Color', [0.85, 0.33, 0.10], 'LineWidth', 1.5, 'DisplayName', 'GAT Train Accuracy');
plot(epochsVec, valAccGAT   * 100, '-',  'Color', [0.85, 0.33, 0.10], 'LineWidth', 2.0, 'DisplayName', 'GAT Val Accuracy');
hold off;
grid on;
xlabel('Epoch', 'FontWeight', 'bold');
ylabel('Accuracy (%)', 'FontWeight', 'bold');
ylim([0 100]);
title('BACE1 Classification: Training vs. Validation Accuracy', 'FontSize', 12);
legend('Location', 'best');

%% --- Local Functions: Enhanced Multi-Head GAT ---
function [loss, grads, probs] = modelLossGAT(params, X, A, graphIdx, Y, numHeads)
    scores = forwardGAT(params, X, A, graphIdx, numHeads);
    probs  = softmax(scores, "DataFormat", "CB");
    weights = dlarray([1.0; 1.5], "CB");
    weightedProbs = probs .* weights;
    loss  = crossentropy(weightedProbs, Y);
    grads = dlgradient(loss, params);
end

function scores = forwardGAT(params, X, A, graphIdx, numHeads)
    H1_att = gatMultiHeadCompute(X, A, params.gat1_W, params.gat1_a, numHeads);
    H1_res = X * params.gat1_Wres;
    H1     = eluActivation(H1_att + H1_res);

    H2_att = gatCompute(H1, A, params.gat2_W, params.gat2_a);
    H2_res = H1 * params.gat2_Wres;
    H2     = eluActivation(H2_att + H2_res);

    H_pool = graphMeanPooling(H2, graphIdx);
    scores = params.fc_W' * H_pool + params.fc_b';
end

function H_out = gatMultiHeadCompute(X, A, W, a, numHeads)
    Z_all = X * W;
    headDim = size(Z_all, 2) / numHeads;
    headOutputs = cell(1, numHeads);

    for h = 1:numHeads
        idxRange = (h-1)*headDim + (1:headDim);
        Z = Z_all(:, idxRange);
        Z1 = Z * a(1:headDim, h);
        Z2 = Z * a(headDim+1:end, h);
        E = leakyrelu(Z1 + Z2', 0.2);
        
        nonNeighborMask = single(A == 0);
        E_masked = E + (-1e9) .* nonNeighborMask;
        Alpha = softmax(E_masked', "DataFormat", "CB")';
        headOutputs{h} = Alpha * Z;
    end
    H_out = cat(2, headOutputs{:});
end

function H_out = gatCompute(X, A, W, a)
    Z = X * W;
    numOutputs = size(Z, 2);
    Z1 = Z * a(1:numOutputs);
    Z2 = Z * a(numOutputs + 1:end);
    E = leakyrelu(Z1 + Z2', 0.2);
    
    nonNeighborMask = single(A == 0);
    E_masked = E + (-1e9) .* nonNeighborMask;
    Alpha = softmax(E_masked', "DataFormat", "CB")';
    H_out = Alpha * Z;
end

function scores = predictGAT(params, graphStructs, batchSize, numHeads)
    numGraphs = numel(graphStructs);
    numClasses = size(params.fc_W, 2);
    scoresAll = zeros(numClasses, numGraphs, "single");
    
    for first = 1:batchSize:numGraphs
        last = min(first + batchSize - 1, numGraphs);
        idx = first:last;
        [A, X, ~, graphIdx] = prepareBatchGAT(graphStructs(idx));
        scoresAll(:, idx) = extractdata(forwardGAT(params, X, A, graphIdx, numHeads));
    end
    scores = dlarray(scoresAll);
end

function [lossVal, accVal] = evaluateMetricsGAT(params, graphStructs, batchSize, numHeads)
    % Compute validation cross-entropy loss and accuracy without gradient tracking
    numGraphs = numel(graphStructs);
    totalLoss = 0;
    totalCorrect = 0;
    for first = 1:batchSize:numGraphs
        last = min(first + batchSize - 1, numGraphs);
        idx = first:last;
        [A, X, Y, graphIdx] = prepareBatchGAT(graphStructs(idx));
        scores = forwardGAT(params, X, A, graphIdx, numHeads);
        probs  = softmax(scores, "DataFormat", "CB");
        loss   = crossentropy(probs, Y, "DataFormat", "CB");
        totalLoss = totalLoss + double(extractdata(loss)) * numel(idx);
        
        [~, predIdx] = max(extractdata(probs), [], 1);
        [~, trueIdx] = max(extractdata(Y), [], 1);
        totalCorrect = totalCorrect + sum(predIdx == trueIdx);
    end
    lossVal = totalLoss / numGraphs;
    accVal  = totalCorrect / numGraphs;
end

function [A_block, X_cat, Y_onehot, graphIdx] = prepareBatchGAT(graphStructs)
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
        adjacency = single(full(graphStructs(i).A) ~= 0);
        adjacency(1:n + 1:end) = 1;
        A_block(idxRange, idxRange) = adjacency;
        X_cat(idxRange, :) = single(full(graphStructs(i).X));
        graphIdx(idxRange) = i;
        if hasLabels
            Y_onehot(double(graphStructs(i).y) + 1, i) = 1;
        end
        currIdx = currIdx + n;
    end
    X_cat    = dlarray(X_cat);
    Y_onehot = dlarray(Y_onehot);
end

%% --- Local Functions: Standard GCN ---
function [loss, grads, probs] = modelLossGCN(params, X, A_norm, graphIdx, Y)
    scores = forwardGCN(params, X, A_norm, graphIdx);
    probs  = softmax(scores, "DataFormat", "CB");
    loss   = crossentropy(probs, Y, "DataFormat", "CB");
    grads  = dlgradient(loss, params);
end

function scores = forwardGCN(params, X, A_norm, graphIdx)
    H1 = eluActivation(A_norm * (X * params.gcn1_W));
    H2 = eluActivation(A_norm * (H1 * params.gcn2_W));
    H_pool = graphMeanPooling(H2, graphIdx);
    scores = params.fc_W' * H_pool + params.fc_b';
end

function scores = predictGCN(params, graphStructs, batchSize)
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

function [lossVal, accVal] = evaluateMetricsGCN(params, graphStructs, batchSize)
    % Compute validation cross-entropy loss and accuracy for GCN
    numGraphs = numel(graphStructs);
    totalLoss = 0;
    totalCorrect = 0;
    for first = 1:batchSize:numGraphs
        last = min(first + batchSize - 1, numGraphs);
        idx = first:last;
        [A_norm, X, Y, graphIdx] = prepareBatchGCN(graphStructs(idx));
        scores = forwardGCN(params, X, A_norm, graphIdx);
        probs = softmax(scores, "DataFormat", "CB");
        loss = crossentropy(probs, Y, "DataFormat", "CB");
        totalLoss = totalLoss + double(extractdata(loss)) * numel(idx);
        
        [~, predIdx] = max(extractdata(probs), [], 1);
        [~, trueIdx] = max(extractdata(Y), [], 1);
        totalCorrect = totalCorrect + sum(predIdx == trueIdx);
    end
    lossVal = totalLoss / numGraphs;
    accVal  = totalCorrect / numGraphs;
end

function [A_norm, X_cat, Y_onehot, graphIdx] = prepareBatchGCN(graphStructs)
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
        A_tilde = single(full(graphStructs(i).A) ~= 0);
        A_tilde(1:n+1:end) = 1;
        deg = sum(A_tilde, 2);
        degInvSqrt = deg .^ (-0.5);
        degInvSqrt(isinf(degInvSqrt) | isnan(degInvSqrt)) = 0;
        A_block(idxRange, idxRange) = diag(degInvSqrt) * A_tilde * diag(degInvSqrt);
        X_cat(idxRange, :) = single(full(graphStructs(i).X));
        graphIdx(idxRange) = i;
        if hasLabels
            Y_onehot(double(graphStructs(i).y) + 1, i) = 1;
        end
        currIdx = currIdx + n;
    end
    A_norm   = dlarray(A_block);
    X_cat    = dlarray(X_cat);
    Y_onehot = dlarray(Y_onehot);
end

%% --- Utility & Metrics Functions ---
function m = computeMetrics(params, testGraphs, batchSize, predictFn)
    scoresTest = predictFn(params, testGraphs, batchSize);
    probsTest  = extractdata(softmax(scoresTest, "DataFormat", "CB"));
    [~, yPredIdx] = max(probsTest, [], 1);
    yPred = reshape(yPredIdx - 1, [], 1);
    yTrue = reshape(double([testGraphs.y]), [], 1);

    C = confusionmat(yTrue, yPred);
    m.TN = C(1,1); m.FP = C(1,2);
    m.FN = C(2,1); m.TP = C(2,2);

    m.acc  = mean(yTrue == yPred);
    m.auc  = binaryAUC(yTrue, probsTest(2, :)');
    m.spec = m.TN / (m.TN + m.FP);
    m.sens = m.TP / (m.TP + m.FN);
    m.npv  = m.TN / (m.TN + m.FN);
    m.ppv  = m.TP / (m.TP + m.FP);
end

function H_pool = graphMeanPooling(H, graphIdx)
    numGraphs = max(graphIdx);
    pooled = cell(1, numGraphs);
    for g = 1:numGraphs
        mask = (graphIdx == g);
        pooled{g} = mean(H(mask, :), 1)';
    end
    H_pool = cat(2, pooled{:});
end

function Y = eluActivation(X)
    Y = max(X, 0) + (exp(min(X, 0)) - 1);
end

function W = initializeGlorot(sz)
    bound = sqrt(6 / sum(sz));
    W = dlarray(single((2 * rand(sz) - 1) * bound));
end

function [tr, va, te] = scaffoldSplit(scaffolds, seed)
    rng(seed, "twister"); 
    key = reshape(string(scaffolds), 1, []); 
    u = unique(key, "stable"); 
    groups = cell(numel(u), 1);
    for k = 1:numel(u)
        groups{k} = reshape(find(key == u(k)), 1, []);
    end
    groups = groups(randperm(numel(groups)));
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
    y = double(y(:)); p = double(p(:));
    n1 = sum(y == 1); n0 = sum(y == 0);
    if n1 == 0 || n0 == 0, auc = NaN; return; end
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
    auc = (sum(ranks(y == 1)) - n1 * (n1 + 1) / 2) / (n1 * n0);
end