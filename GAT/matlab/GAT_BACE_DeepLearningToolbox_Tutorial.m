%% BACE Molecular Graph Classification with a Custom GAT
% Companion script to prepare_bace_graphs.py.
%
% Uses learnable dlarray parameters, dlfeval, and adamupdate.
% This implementation does not construct a dlnetwork object.
%
% Expected graph fields:
%   X        : numNodes-by-numFeatures node features
%   A        : numNodes-by-numNodes adjacency matrix
%   y        : binary label, 0 or 1 (required for BACE graphs)
%   scaffold : scaffold identifier (required for BACE graphs)
%
% Expected additional DrugBank fields:
%   DrugBankID
%   Name
%
% Attention is dense within each minibatch. Reduce batchSize if memory
% remains insufficient.

clear;
clc;
close all;

%% 1. Load Preprocessed Graph Data
matFile = "bace_graphs.mat";

loaded = load(matFile, "graphs", "drugbank");
graphs = loaded.graphs;
drugbank = loaded.drugbank;

assert(~isempty(graphs), "The BACE graph dataset is empty.");

fprintf("Loaded %d graphs; active fraction %.3f.\n", ...
    numel(graphs), mean(double([graphs.y])));

%% 2. Bemis-Murcko Scaffold Split (Approximately 80/10/10)
[trainIdx, valIdx, testIdx] = scaffoldSplit({graphs.scaffold}, 7);

fprintf("Scaffold split: train %d, validation %d, test %d.\n", ...
    numel(trainIdx), numel(valIdx), numel(testIdx));

trainGraphs = graphs(trainIdx);
valGraphs   = graphs(valIdx);
testGraphs  = graphs(testIdx);

assert(~isempty(trainGraphs), "Training split is empty.");
assert(~isempty(valGraphs), ...
    "Validation split is empty. Adjust the scaffold split.");
assert(~isempty(testGraphs), ...
    "Test split is empty. Adjust the scaffold split.");

%% 3. Model Dimensions
numFeatures = size(graphs(1).X, 2);
numClasses = 2;
hiddenUnits = 32;

%% 4. Initialize Learnable Parameters
params = struct();

params.gat1_W = initializeGlorot([numFeatures, hiddenUnits]);
params.gat1_a = initializeGlorot([2 * hiddenUnits, 1]);

params.gat2_W = initializeGlorot([hiddenUnits, hiddenUnits]);
params.gat2_a = initializeGlorot([2 * hiddenUnits, 1]);

params.fc_W = initializeGlorot([hiddenUnits, numClasses]);
params.fc_b = dlarray(zeros(1, numClasses, "single"));

%% 5. Train Using Minibatches
maxEpochs = 100;
batchSize = 16;  % Reduce to 8 or 4 if memory is insufficient.
lr = 0.003;

avgGradients = [];
avgSqGradients = [];
iteration = 0;

fprintf("\n--- Training GAT Architecture ---\n");

for epoch = 1:maxEpochs
    order = randperm(numel(trainGraphs));
    totalTrainLoss = 0;

    for first = 1:batchSize:numel(order)
        last = min(first + batchSize - 1, numel(order));
        batchIdx = order(first:last);

        [A, X, Y, graphIdx] = ...
            prepareBatchData(trainGraphs(batchIdx));

        [loss, grads] = dlfeval( ...
            @modelLoss, params, X, A, graphIdx, Y);

        % Adam iteration increases after every minibatch.
        iteration = iteration + 1;

        [params, avgGradients, avgSqGradients] = adamupdate( ...
            params, grads, avgGradients, avgSqGradients, ...
            iteration, lr);

        totalTrainLoss = totalTrainLoss + ...
            double(extractdata(loss)) * numel(batchIdx);
    end

    trainLoss = totalTrainLoss / numel(trainGraphs);

    if mod(epoch, 10) == 0 || epoch == 1
        valLoss = evaluateLossBatched( ...
            params, valGraphs, batchSize);

        fprintf( ...
            "Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f\n", ...
            epoch, maxEpochs, trainLoss, valLoss);
    end
end

% Release the final training batch before evaluation.
clear A X Y graphIdx loss grads;

%% 6. Test Evaluation and Confusion Matrix
scoresTest = predictGraphsBatched(params, testGraphs, batchSize);

probsTest = extractdata( ...
    softmax(scoresTest, "DataFormat", "CB"));

[~, yPredIdx] = max(probsTest, [], 1);

yPredNum = reshape(yPredIdx - 1, [], 1);
yTrueNum = reshape(double([testGraphs.y]), [], 1);

yPred = categorical(yPredNum, [0 1]);
YTestCat = categorical(yTrueNum, [0 1]);

figure("Color", "w");

confusionchart(YTestCat, yPred, ...
    "Title", "GAT Model — BACE Scaffold Test Set", ...
    "RowSummary", "row-normalized", ...
    "ColumnSummary", "column-normalized");

yProbActive = probsTest(2, :)';

testAcc = mean(yTrueNum == yPredNum);
testAUC = binaryAUC(yTrueNum, yProbActive);

fprintf("\nTest Accuracy: %.3f | ROC-AUC: %.3f (n=%d)\n", ...
    testAcc, testAUC, numel(testGraphs));

%% 7. Inference on External DrugBank Molecules
if isempty(drugbank)
    fprintf("\nNo DrugBank molecules were provided.\n");
else
    scoresDB = predictGraphsBatched( ...
        params, drugbank, batchSize);

    probsDB = extractdata( ...
        softmax(scoresDB, "DataFormat", "CB"));

    probActiveDB = probsDB(2, :)';

    external = table( ...
        reshape(string({drugbank.DrugBankID}), [], 1), ...
        reshape(string({drugbank.Name}), [], 1), ...
        probActiveDB, ...
        'VariableNames', ...
        {'DrugBankID', 'Name', 'PredictedActiveProbability'});

    disp(external);
end

%% --- Local Model and Utility Functions ---

function [loss, grads] = modelLoss(params, X, A, graphIdx, Y)
    scores = forwardGNN(params, X, A, graphIdx);

    probabilities = softmax(scores, "DataFormat", "CB");

    loss = crossentropy( ...
        probabilities, Y, "DataFormat", "CB");

    grads = dlgradient(loss, params);
end

function scores = forwardGNN(params, X, A, graphIdx)
    % GAT Layer 1 + ELU
    H1 = gatLayerCompute( ...
        X, A, params.gat1_W, params.gat1_a);

    H1 = eluActivation(H1);

    % GAT Layer 2 + ELU
    H2 = gatLayerCompute( ...
        H1, A, params.gat2_W, params.gat2_a);

    H2 = eluActivation(H2);

    % hiddenUnits-by-numGraphs
    H_pool = graphMeanPooling(H2, graphIdx);

    % numClasses-by-numGraphs
    scores = params.fc_W' * H_pool + params.fc_b';
end

function H_out = gatLayerCompute(X, A, W, a)
    % X: numNodes-by-numInputFeatures
    % W: numInputFeatures-by-numOutputFeatures
    Z = X * W;

    numOutputs = size(Z, 2);

    a1 = a(1:numOutputs);
    a2 = a(numOutputs + 1:end);

    Z1 = Z * a1;
    Z2 = Z * a2;

    % Pairwise attention logits.
    E = leakyrelu(Z1 + Z2', 0.2);

    % A is ordinary numeric data, not a learnable dlarray.
    nonNeighborMask = single(A == 0);

    E_masked = E + (-1e9) .* nonNeighborMask;

    % Softmax with CB normalizes the first dimension.
    % Transposing makes normalization occur over each row's neighbors.
    Alpha = softmax(E_masked', "DataFormat", "CB")';

    H_out = Alpha * Z;
end

function H_pool = graphMeanPooling(H, graphIdx)
    numGraphs = max(graphIdx);

    % Concatenate dlarray results to preserve the gradient trace.
    pooled = cell(1, numGraphs);

    for g = 1:numGraphs
        mask = (graphIdx == g);
        pooled{g} = mean(H(mask, :), 1)';
    end

    H_pool = cat(2, pooled{:});
end

function Y = eluActivation(X)
    % ELU with alpha = 1.
    % Clamp the exponent's input to avoid evaluating exp on large
    % positive values.
    Y = max(X, 0) + (exp(min(X, 0)) - 1);
end

function W = initializeGlorot(sz)
    bound = sqrt(6 / sum(sz));

    W = dlarray(single( ...
        (2 * rand(sz) - 1) * bound));
end

function [A_block, X_cat, Y_onehot, graphIdx] = ...
        prepareBatchData(graphStructs)

    numGraphs = numel(graphStructs);

    assert(numGraphs > 0, "Cannot prepare an empty minibatch.");

    numNodesTotal = sum(arrayfun( ...
        @(g) size(g.X, 1), graphStructs));

    numFeatures = size(graphStructs(1).X, 2);

    % Adjacency remains ordinary numeric data.
    A_block = zeros(numNodesTotal, numNodesTotal, "single");

    X_cat = zeros(numNodesTotal, numFeatures, "single");
    graphIdx = zeros(numNodesTotal, 1);

    hasLabels = isfield(graphStructs, "y");

    if hasLabels
        Y_onehot = zeros(2, numGraphs, "single");
    else
        Y_onehot = [];
    end

    currIdx = 0;

    for i = 1:numGraphs
        n = size(graphStructs(i).X, 1);

        assert(n > 0, "Each graph must contain at least one node.");
        assert(size(graphStructs(i).X, 2) == numFeatures, ...
            "All graphs must have the same number of features.");
        assert(isequal(size(graphStructs(i).A), [n n]), ...
            "Graph adjacency dimensions must match its node count.");

        idxRange = currIdx + (1:n);

        % Treat nonzero adjacency entries as edges and add self-loops.
        adjacency = single(full(graphStructs(i).A) ~= 0);
        adjacency(1:n + 1:end) = 1;

        A_block(idxRange, idxRange) = adjacency;
        X_cat(idxRange, :) = single(full(graphStructs(i).X));
        graphIdx(idxRange) = i;

        if hasLabels
            label = double(graphStructs(i).y);

            assert(isscalar(label) && ...
                ismember(label, [0 1]), ...
                "Graph labels must be scalar 0 or 1.");

            Y_onehot(label + 1, i) = 1;
        end

        currIdx = currIdx + n;
    end

    X_cat = dlarray(X_cat);

    if hasLabels
        Y_onehot = dlarray(Y_onehot);
    end
end

function meanLoss = evaluateLossBatched( ...
        params, graphStructs, batchSize)

    numGraphs = numel(graphStructs);

    assert(numGraphs > 0, ...
        "Cannot evaluate an empty graph set.");

    totalLoss = 0;

    for first = 1:batchSize:numGraphs
        last = min(first + batchSize - 1, numGraphs);
        idx = first:last;

        [A, X, Y, graphIdx] = ...
            prepareBatchData(graphStructs(idx));

        scores = forwardGNN(params, X, A, graphIdx);

        probabilities = softmax( ...
            scores, "DataFormat", "CB");

        % Validation does not compute gradients.
        loss = crossentropy( ...
            probabilities, Y, "DataFormat", "CB");

        totalLoss = totalLoss + ...
            double(extractdata(loss)) * numel(idx);
    end

    meanLoss = totalLoss / numGraphs;
end

function scores = predictGraphsBatched( ...
        params, graphStructs, batchSize)

    numGraphs = numel(graphStructs);
    numClasses = size(params.fc_W, 2);

    scoresAll = zeros(numClasses, numGraphs, "single");

    for first = 1:batchSize:numGraphs
        last = min(first + batchSize - 1, numGraphs);
        idx = first:last;

        [A, X, ~, graphIdx] = ...
            prepareBatchData(graphStructs(idx));

        batchScores = forwardGNN(params, X, A, graphIdx);

        scoresAll(:, idx) = extractdata(batchScores);
    end

    scores = dlarray(scoresAll);
end

function [tr, va, te] = scaffoldSplit(scaffolds, seed)
    rng(seed, "twister");

    % Force a row vector so group concatenation is consistent.
    key = reshape(string(scaffolds), 1, []);

    u = unique(key, "stable");
    groups = cell(numel(u), 1);

    for k = 1:numel(u)
        groups{k} = reshape(find(key == u(k)), 1, []);
    end

    groups = groups(randperm(numel(groups)));

    n = numel(key);
    tr = [];
    va = [];
    te = [];

    % Whole scaffold groups are kept together.
    % Group sizes can cause the target proportions to be exceeded.
    for k = 1:numel(groups)
        g = groups{k};

        if numel(tr) < 0.8 * n
            tr = [tr, g]; %#ok<AGROW>
        elseif numel(va) < 0.1 * n
            va = [va, g]; %#ok<AGROW>
        else
            te = [te, g]; %#ok<AGROW>
        end
    end
end

function auc = binaryAUC(y, p)
    y = double(y(:));
    p = double(p(:));

    assert(numel(y) == numel(p), ...
        "Labels and probabilities must have matching lengths.");

    n1 = sum(y == 1);
    n0 = sum(y == 0);

    if n1 == 0 || n0 == 0
        auc = NaN;
        return;
    end

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

    auc = (sum(ranks(y == 1)) - n1 * (n1 + 1) / 2) ...
        / (n1 * n0);
end