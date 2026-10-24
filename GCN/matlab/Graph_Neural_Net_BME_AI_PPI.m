%% Protein Function Prediction on PPI Networks
% MATLAB Live Script adaptation comparing an MLP and a GCN model
% on tissue-specific protein-protein interaction (PPI) graphs.

clear; clc; close all

%% 1. Load PPI graphs and inspect a tissue
load("ppi.mat");

g = trainGraphs{1}; 
X = single(g.X); 
Y = single(g.Y); 
A = spones(sparse(g.A));

[nNodes, nFeatures] = size(X); 
nLabels = size(Y, 2);

fprintf("Example tissue: %d proteins, %d features, %d GO terms\n", nNodes, nFeatures, nLabels);
fprintf("Mean interactions/protein: %.1f | Mean GO terms/protein: %.1f\n", nnz(A)/nNodes, mean(sum(Y,2)));

%% 2. Visualize GO-term sharing among interacting proteins
prevalence = mean(Y, 1); 
[~, goTerm] = min(abs(prevalence - 0.30));

[i, j] = find(triu(A, 1));
connected = mean(Y(i,:) .* Y(j,:), 1);

rng(0); 
ri = randi(nNodes, numel(i), 1); 
rj = randi(nNodes, numel(i), 1);
randomPairs = mean(Y(ri,:) .* Y(rj,:), 1);

figure("Color", "w"); 
scatter(randomPairs, connected, 18, "filled"); 
hold on
lim = max([connected randomPairs]); 
plot([0 lim], [0 lim], "k--"); 
axis square
xlabel("Both proteins have GO term (random pairs)"); 
ylabel("Both proteins have GO term (interacting pairs)");
title("GO-term sharing across protein pairs");

fprintf("%.1f%% of terms are shared more often by interacting proteins.\n", 100 * mean(connected > randomPairs));

%% 3. Compare MLP and GCN models
hiddenSize = 128; 
epochs = 100; 
learnRate = 0.005;

mlp = initializeModel(nFeatures, hiddenSize, nLabels);
gcn = initializeModel(nFeatures, hiddenSize, nLabels);

[mlp, mlpHistory] = trainModel(mlp, trainGraphs, valGraphs, epochs, learnRate, false);
[gcn, gcnHistory] = trainModel(gcn, trainGraphs, valGraphs, epochs, learnRate, true);

%% 4. Compare training curves
figure("Color", "w"); 
tiledlayout(1, 2);

nexttile; 
plot(mlpHistory.trainF1); hold on; plot(mlpHistory.valF1); 
title("MLP"); xlabel("Epoch"); ylabel("Micro-F1"); 
legend("Train", "Validation"); ylim([0 1]); grid on

nexttile; 
plot(gcnHistory.trainF1); hold on; plot(gcnHistory.valF1); 
title("GCN"); xlabel("Epoch"); ylabel("Micro-F1"); 
legend("Train", "Validation"); ylim([0 1]); grid on

%% 5. Evaluate on unseen tissue graphs
mlpF1 = evaluateGraphs(mlp, testGraphs, false);
gcnF1 = evaluateGraphs(gcn, testGraphs, true);

fprintf("\nFinal Results:\n");
fprintf("MLP test micro-F1: %.3f\n", mlpF1);
fprintf("GCN test micro-F1: %.3f\n", gcnF1);

%% 6. Compare test scores
figure("Color", "w"); 
bar([mlpF1 gcnF1]); 
xticklabels(["MLP", "GCN"]); 
ylabel("Test micro-F1"); 
ylim([0 1]); 
title("Protein function prediction on held-out tissues"); 
grid on

%% 7. Inspect learned hidden representations
t = testGraphs{1}; 
[~, Hmlp] = forwardModel(mlp, single(t.X), sparse(size(t.X,1), size(t.X,1)), false);
[~, Hgcn] = forwardModel(gcn, single(t.X), normalizeAdj(t.A), true);

figure("Color", "w"); 
tiledlayout(1, 2);

nexttile; 
Z1 = tsne(Hmlp'); 
gscatter(Z1(:,1), Z1(:,2), logical(t.Y(:, goTerm))); 
title("MLP hidden features");

nexttile; 
Z2 = tsne(Hgcn'); 
gscatter(Z2(:,1), Z2(:,2), logical(t.Y(:, goTerm))); 
title("GCN hidden features");

%% Local Functions

function m = initializeModel(nIn, nHidden, nOut)
    s = 0.02;
    m.W1 = single(randn(nHidden, nIn) * s);
    m.b1 = zeros(nHidden, 1, "single");
    m.W2 = single(randn(nOut, nHidden) * s);
    m.b2 = zeros(nOut, 1, "single");
end

function A = normalizeAdj(A)
    A = spones(sparse(A));
    A = A + A' + speye(size(A, 1));
    d = sum(A, 2);
    d_inv_sqrt = 1 ./ sqrt(d);
    d_inv_sqrt(isinf(d_inv_sqrt) | isnan(d_inv_sqrt)) = 0;
    D_mat = spdiags(d_inv_sqrt, 0, numel(d), numel(d));
    A = D_mat * A * D_mat;
end

function [out, H] = forwardModel(m, X, A, useGraph)
    X = X'; 
    A = single(normalizeAdj(A));
    
    if useGraph
        Z = m.W1 * (X * A) + m.b1;
    else
        Z = m.W1 * X + m.b1;
    end
    
    H = max(Z, 0); % ReLU activation
    
    if useGraph
        out = m.W2 * (H * A) + m.b2;
    else
        out = m.W2 * H + m.b2;
    end
end

function [m, hist] = trainModel(m, trainG, valG, epochs, lr, useGraph)
    hist.trainF1 = zeros(epochs, 1);
    hist.valF1   = zeros(epochs, 1);
    
    beta1 = 0.9; 
    beta2 = 0.999; 
    epsAdam = 1e-8;
    
    v = struct(); 
    s = struct();
    names = fieldnames(m);
    for z = 1:numel(names)
        v.(names{z}) = zeros(size(m.(names{z})), "single");
        s.(names{z}) = zeros(size(m.(names{z})), "single");
    end
    
    t_step = 0;
    for ep = 1:epochs
        order = randperm(numel(trainG));
        for q = order
            t_step = t_step + 1;
            g = trainG{q}; 
            X = single(g.X); 
            Y = single(g.Y); 
            A = sparse(g.A);
            
            [loss, gr] = graphGrad(m, X, Y, A, useGraph);
            
            for z = 1:numel(names)
                nm = names{z};
                v.(nm) = beta1 * v.(nm) + (1 - beta1) * gr.(nm);
                s.(nm) = beta2 * s.(nm) + (1 - beta2) * (gr.(nm).^2);
                
                v_hat = v.(nm) / (1 - beta1^t_step);
                s_hat = s.(nm) / (1 - beta2^t_step);
                
                m.(nm) = m.(nm) - lr * v_hat ./ (sqrt(s_hat) + epsAdam);
            end
        end
        
        hist.trainF1(ep) = evaluateGraphs(m, trainG, useGraph);
        hist.valF1(ep)   = evaluateGraphs(m, valG, useGraph);
        
        if mod(ep, 10) == 0
            fprintf("Epoch %3d | Loss: %.4f | Train F1: %.3f | Val F1: %.3f\n", ...
                ep, loss, hist.trainF1(ep), hist.valF1(ep));
        end
    end
end

function [loss, gr] = graphGrad(m, X, Y, A, useGraph)
    A = single(normalizeAdj(A));
    X = X';
    
    if useGraph
        XA = X * A;
    else
        XA = X;
    end
    
    Z = m.W1 * XA + m.b1;
    H = max(Z, 0);
    
    if useGraph
        HA = H * A;
    else
        HA = H;
    end
    
    O = m.W2 * HA + m.b2;
    P = 1 ./ (1 + exp(-max(-30, min(30, O))));
    T = Y';
    
    loss = mean(max(O, 0) - O .* T + log1p(exp(-abs(O))), "all");
    
    dO = (P - T) / numel(T);
    dW2 = dO * HA';
    db2 = sum(dO, 2);
    
    dH = m.W2' * dO;
    if useGraph
        dH = dH * A;
    end
    
    dZ = dH .* (Z > 0);
    dW1 = dZ * XA';
    db1 = sum(dZ, 2);
    
    gr = struct("W1", dW1, "b1", db1, "W2", dW2, "b2", db2);
end

function f = evaluateGraphs(m, G, useGraph)
    TP = 0; FP = 0; FN = 0;
    for k = 1:numel(G)
        g = G{k}; 
        [O, ~] = forwardModel(m, single(g.X), sparse(g.A), useGraph);
        P = O' > 0;
        T = logical(g.Y);
        
        TP = TP + nnz(P & T);
        FP = FP + nnz(P & ~T);
        FN = FN + nnz(~P & T);
    end
    f = 2 * TP / max(2 * TP + FP + FN, 1);
end