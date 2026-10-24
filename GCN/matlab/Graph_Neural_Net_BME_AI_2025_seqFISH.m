%% Graph Neural Network for Cell Type Classification (seqFISH)
clear; clc; close all

%% 1. Load and inspect the data
dataFile = "seqfish.mat";
S = load(dataFile);

X = single(S.X);
xy = double(S.xy);
cellType = string(S.cellType(:));
assert(size(X,1) == size(xy,1) && size(X,1) == numel(cellType), ...
    "Rows must refer to the same cells.");

y = categorical(cellType);
cellNames = categories(y);
numNodes = size(X,1);
numFeatures = size(X,2);
numClasses = numel(cellNames);
fprintf("%d cells, %d genes, %d cell types\n", numNodes, numFeatures, numClasses);

%% 2. Visualize spatial distribution
figure("Color","w");
gscatter(xy(:,1), xy(:,2), y, [], [], 8);
set(gca, "YDir", "reverse");
axis equal tight
xlabel("Spatial coordinate 1");
ylabel("Spatial coordinate 2");
title("seqFISH cells by mapped cell type");
legend("Location", "eastoutside");

%% 3. Build normalized adjacency matrix
neighborCount = 8;
idx = knnsearch(xy, xy, "K", neighborCount + 1);
src = repelem((1:numNodes)', neighborCount);
dst = reshape(idx(:,2:end), [], 1);
A = sparse(src, dst, 1, numNodes, numNodes);
A = spones(A + A');
A = A + speye(numNodes);

D = sum(A, 2);
Ahat = spdiags(1./sqrt(D), 0, numNodes, numNodes) * A * spdiags(1./sqrt(D), 0, numNodes, numNodes);

figure("Color","w");
nPlot = min(500, numNodes);
gplot(A(1:nPlot, 1:nPlot), xy(1:nPlot,:), "-");
hold on;
scatter(xy(:,1), xy(:,2), 5, double(y), "filled");
axis equal tight
subtitle("Graph edges among first 500 cells");

%% 4. Stratified train/test split
rng(7);
trainMask = false(numNodes, 1);
testMask = false(numNodes, 1);
for k = 1:numClasses
    members = find(y == cellNames{k});
    members = members(randperm(numel(members)));
    nTest = max(1, round(0.2 * numel(members)));
    testMask(members(1:nTest)) = true;
    trainMask(members(nTest+1:end)) = true;
end

%% 5. Define GCN architecture using dlnetwork
hiddenSize = 64;

layers = [
    featureInputLayer(numFeatures, "Name", "input", "Normalization", "none")
    graphConvLayer(numFeatures, hiddenSize, Ahat, "gcn1")
    reluLayer("Name", "relu1")
    dropoutLayer(0.3, "Name", "drop1")
    graphConvLayer(hiddenSize, numClasses, Ahat, "gcn2")
    softmaxLayer("Name", "softmax")
];

net = dlnetwork(layers);

%% 6. Train the GCN network
epochs = 200;
learnRate = 0.01;
weightDecay = 5e-4;

trailingAvg = [];
trailingAvgSq = [];
lossHistory = zeros(epochs, 1);
accuracyHistory = zeros(epochs, 1);

Xdl = dlarray(X', "CB");

for epoch = 1:epochs
    [loss, gradients] = dlfeval(@modelLoss, net, Xdl, y, trainMask, weightDecay);
    [net, trailingAvg, trailingAvgSq] = adamupdate( ...
        net, gradients, trailingAvg, trailingAvgSq, epoch, learnRate);
        
    lossHistory(epoch) = double(extractdata(loss));
    
    scores = predict(net, Xdl);
    [~, predIdx] = max(extractdata(scores), [], 1);
    pred = categorical(predIdx', 1:numClasses, cellNames);
    accuracyHistory(epoch) = mean(pred(testMask) == y(testMask));
    
    if mod(epoch, 20) == 0
        fprintf("Epoch %3d | Loss: %.4f | Test Accuracy: %.3f\n", ...
            epoch, lossHistory(epoch), accuracyHistory(epoch));
    end
end

%% 7. Evaluate and plot results
scores = predict(net, Xdl);
[~, predIdx] = max(extractdata(scores), [], 1);
pred = categorical(predIdx', 1:numClasses, cellNames);
acc = mean(pred(testMask) == y(testMask));
fprintf("Final Test Accuracy: %.3f\n", acc);

figure("Color","w");
tiledlayout(1,2);
nexttile;
plot(lossHistory, "LineWidth", 1.5);
xlabel("Epoch"); ylabel("Loss"); title("Training Loss"); grid on;
nexttile;
plot(accuracyHistory, "LineWidth", 1.5);
xlabel("Epoch"); ylabel("Accuracy"); title("Test Accuracy");
ylim([0 1]); grid on;

cm = confusionmat(y(testMask), pred(testMask), "Order", categorical(cellNames));
cm = 100 * cm ./ max(sum(cm, 2), 1);
figure("Color","w");
heatmap(cellNames, cellNames, cm, "ColorLimits", [0 100]);
xlabel("Predicted Cell Type");
ylabel("True Cell Type");
title("Row-Normalized Confusion Matrix (%)");

%% Helper Functions
function [loss, gradients] = modelLoss(net, X, y, mask, weightDecay)
scores = forward(net, X);
scoresMasked = scores(:, mask);

% Convert categorical labels for masked nodes into one-hot target matrix [Classes x Nodes]
labelsMasked = y(mask);
numClasses = size(scores, 1);
numMasked = numel(labelsMasked);

T = zeros(numClasses, numMasked, "single");
T(sub2ind([numClasses, numMasked], double(labelsMasked)', 1:numMasked)) = 1;
targetsMasked = dlarray(T, "CB");

% Compute cross-entropy loss between formatted dlarrays
loss = crossentropy(scoresMasked, targetsMasked);

% L2 Regularization over network parameters
for i = 1:height(net.Learnables)
    loss = loss + weightDecay * sum(net.Learnables.Value{i}.^2, "all");
end

gradients = dlgradient(loss, net.Learnables);
end