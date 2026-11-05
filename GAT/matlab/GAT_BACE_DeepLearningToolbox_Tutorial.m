%% BACE Molecular Graph Classification with dlnetwork
% Companion script to prepare_bace_graphs.py.
% Constructs a custom GAT architecture using dlnetwork and local layer functions,
% trains via dlfeval/adamupdate, predicts DrugBank molecules, and evaluates test set performance.
clear; clc; close all;

%% 1. Load Preprocessed Graph Data
matFile = "bace_graphs.mat";

loaded = load(matFile, "graphs", "drugbank");
graphs = loaded.graphs; 
drugbank = loaded.drugbank;

fprintf("Loaded %d graphs; active fraction %.3f.\n", numel(graphs), mean([graphs.y]));

%% 2. Bemis-Murcko Scaffold Split (approx 80/10/10)
[trainIdx, valIdx, testIdx] = scaffoldSplit({graphs.scaffold}, 7);

fprintf("Scaffold split: train %d, validation %d, test %d.\n", ...
    numel(trainIdx), numel(valIdx), numel(testIdx));

trainGraphs = graphs(trainIdx);
valGraphs   = graphs(valIdx);
testGraphs  = graphs(testIdx);

%% 3. Prepare Batch Data (Block-Diagonal Adjacency & Concatenated Features)
[ATrain, XTrain, YTrain, graphIdxTrain] = prepareBatchData(trainGraphs);
[AVal,   XVal,   YVal,   graphIdxVal]   = prepareBatchData(valGraphs);
[ATest,  XTest,  YTest,  graphIdxTest]  = prepareBatchData(testGraphs);

numFeatures = size(graphs(1).X, 2);
numClasses  = 2;
hiddenUnits = 32;

%% 4. Define and Initialize dlnetwork Architecture & Parameters
% Feature input layer definition
inputLayer = featureInputLayer(numFeatures, "Normalization", "none", "Name", "ginput");

% Initialize Learnable Parameters
params = struct();
params.gat1_W = initializeGlorot([numFeatures, hiddenUnits]);
params.gat1_a = initializeGlorot([2 * hiddenUnits, 1]);
params.gat2_W = initializeGlorot([hiddenUnits, hiddenUnits]);
params.gat2_a = initializeGlorot([2 * hiddenUnits, 1]);
params.fc_W   = initializeGlorot([hiddenUnits, numClasses]);
params.fc_b   = zeros(1, numClasses, "single");

%% 5. Train GAT Architecture
maxEpochs = 100;
lr = 0.003;
avgGradients = [];
avgSqGradients = [];

fprintf("\n--- Training GAT Architecture ---\n");
for epoch = 1:maxEpochs
    [loss, grads] = dlfeval(@modelLoss, params, XTrain, ATrain, graphIdxTrain, YTrain);
    
    % Update learnable parameters using Adam optimizer
    [params, avgGradients, avgSqGradients] = adamupdate(params, grads, ...
        avgGradients, avgSqGradients, epoch, lr);
    
    if mod(epoch, 10) == 0 || epoch == 1
        valLoss = modelLoss(params, XVal, AVal, graphIdxVal, YVal);
        fprintf("Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f\n", ...
            epoch, maxEpochs, extractdata(loss), extractdata(valLoss));
    end
end

%% 6. Test Evaluation & Confusion Matrix
scoresTest = forwardGNN(params, XTest, ATest, graphIdxTest);
probsTest = extractdata(softmax(scoresTest, "DataFormat", "CB"));
[~, yPredIdx] = max(probsTest, [], 1);
yPred = categorical(yPredIdx - 1);

% Convert target labels to categorical
[~, trueIdx] = max(extractdata(YTest), [], 1);
YTestCat = categorical(trueIdx - 1);

figure("Color", "w");
confusionchart(YTestCat, yPred, ...
    "Title", "GAT Model — BACE Scaffold Test Set", ...
    "RowSummary", "row-normalized", ...
    "ColumnSummary", "column-normalized");

yTrueNum = double(YTestCat == '1');
yProbActive = probsTest(2, :)';
testAcc = mean(YTestCat == yPred);
testAUC = binaryAUC(yTrueNum, yProbActive);
fprintf("\nTest Accuracy: %.3f | ROC-AUC: %.3f (n=%d)\n", testAcc, testAUC, numel(testGraphs));

%% 7. Inference on External DrugBank Molecules
[ADrugbank, XDrugbank, ~, graphIdxDB] = prepareBatchData(drugbank);
scoresDB = forwardGNN(params, XDrugbank, ADrugbank, graphIdxDB);
probsDB = extractdata(softmax(scoresDB, "DataFormat", "CB"));
probActiveDB = probsDB(2, :)';

external = table(string({drugbank.DrugBankID})', string({drugbank.Name})', probActiveDB, ...
    'VariableNames', {'DrugBankID', 'Name', 'PredictedActiveProbability'});
disp(external);


%% --- Local Model & Layer Functions ---

function [loss, grads] = modelLoss(params, X, A, graphIdx, Y)
    scores = forwardGNN(params, X, A, graphIdx);
    loss = crossentropy(softmax(scores, "DataFormat", "CB"), Y, "DataFormat", "CB");
    grads = dlgradient(loss, params);
end

function scores = forwardGNN(params, X, A, graphIdx)
    % GAT Layer 1 + ELU
    H1 = gatLayerCompute(X, A, params.gat1_W, params.gat1_a);
    H1 = elu(H1);
    
    % GAT Layer 2 + ELU
    H2 = gatLayerCompute(H1, A, params.gat2_W, params.gat2_a);
    H2 = elu(H2);
    
    % Global Mean Pooling
    H_pool = graphMeanPooling(H2, graphIdx);
    
    % Output Linear Layer
    scores = params.fc_W' * H_pool + params.fc_b';
end

function H_out = gatLayerCompute(X, A, W, a)
    Z = X * W;
    numOutputs = size(Z, 2);
    
    a1 = a(1:numOutputs);
    a2 = a(numOutputs+1:end);
    
    Z1 = Z * a1;
    Z2 = Z * a2;
    E = leakyrelu(Z1 + Z2', 0.2);
    
    E_masked = E;
    E_masked(A == 0) = -1e9;
    Alpha = softmax(E_masked, "DataFormat", "CB");
    
    H_out = Alpha * Z;
end

function H_pool = graphMeanPooling(H, graphIdx)
    numGraphs = max(graphIdx);
    hiddenDim = size(H, 2);
    H_pool = dlarray(zeros(hiddenDim, numGraphs, "single"));
    
    for g = 1:numGraphs
        mask = (graphIdx == g);
        H_pool(:, g) = mean(H(mask, :), 1)';
    end
end

function W = initializeGlorot(sz)
    bound = sqrt(6 / sum(sz));
    W = dlarray(single((2 * rand(sz) - 1) * bound));
end

function [A_block, X_cat, Y_onehot, graphIdx] = prepareBatchData(graphStructs)
    numGraphs = numel(graphStructs);

    if numGraphs == 0
        A_block = dlarray(zeros(0, 0, "single"));
        X_cat = dlarray(zeros(0, 0, "single"));
        Y_onehot = dlarray(zeros(2, 0, "single"));
        graphIdx = zeros(0, 1);
        return;
    end

    numNodesTotal = sum(arrayfun(@(g) size(g.X, 1), graphStructs));
    numFeatures = size(graphStructs(1).X, 2);
    
    A_block = zeros(numNodesTotal, numNodesTotal, "single");
    X_cat   = zeros(numNodesTotal, numFeatures, "single");
    graphIdx = zeros(numNodesTotal, 1);
    
    if isfield(graphStructs, "y")
        Y_onehot = zeros(2, numGraphs, "single");
    else
        Y_onehot = [];
    end
    
    currIdx = 0;
    for i = 1:numGraphs
        n = size(graphStructs(i).X, 1);
        idxRange = currIdx + (1:n);
        
        A_block(idxRange, idxRange) = graphStructs(i).A + eye(n, "single"); 
        X_cat(idxRange, :) = graphStructs(i).X;
        graphIdx(idxRange) = i;
        
        if isfield(graphStructs, "y")
            label = graphStructs(i).y;
            Y_onehot(label + 1, i) = 1;
        end
        
        currIdx = currIdx + n;
    end
    
    A_block  = dlarray(A_block);
    X_cat    = dlarray(X_cat);
    Y_onehot = dlarray(Y_onehot);
end

function [tr, va, te] = scaffoldSplit(scaffolds, seed)
    rng(seed, "twister");
    key = string(scaffolds);
    u = unique(key, "stable");
    groups = cell(numel(u), 1);
    for k = 1:numel(u)
        groups{k} = find(key == u(k));
    end
    groups = groups(randperm(numel(groups)));
    n = numel(key); 
    tr = []; va = []; te = [];
    for k = 1:numel(groups)
        g = groups{k};
        if numel(tr) < 0.8 * n
            tr = [tr, g]; %#ok<AGROW>
        elseif numel(va) < 0.9 * n
            va = [va, g]; %#ok<AGROW>
        else
            te = [te, g]; %#ok<AGROW>
        end
    end
end

function a = binaryAUC(y, p)
    n1 = sum(y == 1);
    n0 = sum(y == 0);
    if n1 == 0 || n0 == 0
        a = NaN;
        return;
    end
    [ps, ord] = sort(p);
    ranks = zeros(size(p));
    i = 1;
    while i <= numel(ps)
        j = i;
        while j < numel(ps) && ps(j+1) == ps(i)
            j = j + 1;
        end
        ranks(ord(i:j)) = mean(i:j);
        i = j + 1;
    end
    a = (sum(ranks(y == 1)) - n1 * (n1 + 1) / 2) / (n1 * n0);
end