%% ========================================================================
%  BACKGROUND & DATASET OVERVIEW: Protein Function Prediction on PPI Networks
%  ========================================================================
%  Biological Context:
%    Proteins rarely act in isolation; instead, they form physical and 
%    functional Protein-Protein Interaction (PPI) networks that drive cellular 
%    processes. Because proteins that physically interact frequently participate 
%    in the same biological pathways, graph structure provides strong predictive 
%    signal for inferring unknown protein functions.
%
%  Dataset Description (Tissue-Specific PPI Benchmark):
%    - Source       : Zitnik & Leskovec (2017) / GraphSAGE Inductive Benchmark.
%    - Structure    : 24 distinct human tissue-specific PPI graphs split into
%                     training (20 tissues), validation (2 tissues), and 
%                     held-out test (2 tissues) graphs (inductive learning).
%    - Nodes (V)    : Individual human proteins (~2,373 proteins per graph).
%    - Edges (E)    : Physical protein-protein interactions within a tissue.
%    - Features (X) : 50-dimensional biological node features (positional gene 
%                     sets, motif gene sets, and immunological signatures).
%    - Labels (Y)   : 121 binary Gene Ontology (GO) cellular function terms 
%                     (multi-label node classification: each protein can 
%                     belong to multiple functional categories simultaneously).
%
%  Model Comparison (MLP vs. GCN):
%    - Multi-Layer Perceptron (MLP): Predicts GO terms solely from a protein's 
%      own 50-dimensional feature vector, ignoring network topology.
%    - Graph Convolutional Network (GCN): Aggregates features across interacting 
%      protein neighborhoods via normalized adjacency matrices, testing how much 
%      network context improves generalization to unseen tissues.
% ========================================================================

% Clear workspace variables, reset command window, and close all figure windows
clear; clc; close all

%% 1. Load PPI graphs and inspect a tissue
% Load pre-split cell arrays: trainGraphs, valGraphs, and testGraphs
load("ppi.mat");

% Inspect the first training tissue graph as a representative example
g = trainGraphs{1}; 
X = single(g.X);            % Node feature matrix [nNodes x 50]
Y = single(g.Y);            % Multi-label binary target matrix [nNodes x 121]
A = spones(sparse(g.A));    % Unweighted sparse adjacency matrix [nNodes x nNodes]

% Extract graph dimensions
[nNodes, nFeatures] = size(X); 
nLabels = size(Y, 2);

% Print summary statistics for the example tissue graph
fprintf("Example tissue: %d proteins, %d features, %d GO terms\n", nNodes, nFeatures, nLabels);
fprintf("Mean interactions/protein: %.1f | Mean GO terms/protein: %.1f\n", nnz(A)/nNodes, mean(sum(Y,2)));

%% 2. Visualize GO-term sharing among interacting proteins
% Compute fraction of proteins annotated with each GO term and pick one near 30% prevalence
prevalence = mean(Y, 1); 
[~, goTerm] = min(abs(prevalence - 0.30));

% Extract all unique interacting protein pairs (upper triangle without self-loops)
[i, j] = find(triu(A, 1));
% Compute probability that both interacting proteins share each GO term
connected = mean(Y(i,:) .* Y(j,:), 1);

% Sample an equal number of random protein pairs as a baseline control
rng(0); 
ri = randi(nNodes, numel(i), 1); 
rj = randi(nNodes, numel(i), 1);
randomPairs = mean(Y(ri,:) .* Y(rj,:), 1);

% Plot co-occurrence rates for interacting pairs vs. random pairs against the y=x diagonal
figure("Color", "w"); 
scatter(randomPairs, connected, 18, "filled"); 
hold on
lim = max([connected randomPairs]); 
plot([0 lim], [0 lim], "k--"); % Points above this line indicate homophily (network enrichment)
axis square
xlabel("Both proteins have GO term (random pairs)"); 
ylabel("Both proteins have GO term (interacting pairs)");
title("GO-term sharing across protein pairs");
fprintf("%.1f%% of terms are shared more often by interacting proteins.\n", 100 * mean(connected > randomPairs));

%% 3. Compare MLP and GCN models
% Define shared architectural and optimization hyperparameters
hiddenSize = 128;   % Hidden layer dimensionality
epochs = 100;       % Number of full passes over the 20 training tissue graphs
learnRate = 0.005;  % Adam optimizer learning rate

% Initialize identical 2-layer architectures for MLP (no graph edges) and GCN (with graph edges)
mlp = initializeModel(nFeatures, hiddenSize, nLabels);
gcn = initializeModel(nFeatures, hiddenSize, nLabels);

% Train both models (useGraph = false for MLP, true for GCN)
[mlp, mlpHistory] = trainModel(mlp, trainGraphs, valGraphs, epochs, learnRate, false);
[gcn, gcnHistory] = trainModel(gcn, trainGraphs, valGraphs, epochs, learnRate, true);

%% 4. Compare training curves
% Plot side-by-side training and validation Micro-F1 learning curves for MLP and GCN
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
% Compute inductive Micro-F1 score across held-out test tissues
mlpF1 = evaluateGraphs(mlp, testGraphs, false);
gcnF1 = evaluateGraphs(gcn, testGraphs, true);

fprintf("\nFinal Results:\n");
fprintf("MLP test micro-F1: %.3f\n", mlpF1);
fprintf("GCN test micro-F1: %.3f\n", gcnF1);

%% 6. Compare test scores
% Bar chart comparing final test Micro-F1 generalization performance
figure("Color", "w"); 
bar([mlpF1 gcnF1]); 
xticklabels(["MLP", "GCN"]); 
ylabel("Test micro-F1"); 
ylim([0 1]); 
title("Protein function prediction on held-out tissues"); 
grid on

%% 7. Inspect learned hidden representations
% Extract 128-dim hidden node embeddings on the first unseen test tissue
t = testGraphs{1}; 
[~, Hmlp] = forwardModel(mlp, single(t.X), sparse(size(t.X,1), size(t.X,1)), false);
[~, Hgcn] = forwardModel(gcn, single(t.X), normalizeAdj(t.A), true);

% Project hidden embeddings into 2D using t-SNE and color by the chosen GO term
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
    % Initialize 2-layer network weights with small Gaussian noise and zero biases
    s = 0.02;
    m.W1 = single(randn(nHidden, nIn) * s);
    m.b1 = zeros(nHidden, 1, "single");
    m.W2 = single(randn(nOut, nHidden) * s);
    m.b2 = zeros(nOut, 1, "single");
end

function A = normalizeAdj(A)
    % Symmetrically normalize adjacency matrix with self-loops: D^(-1/2) * (A + A' + I) * D^(-1/2)
    A = spones(sparse(A));
    A = A + A' + speye(size(A, 1));
    d = sum(A, 2);
    d_inv_sqrt = 1 ./ sqrt(d);
    d_inv_sqrt(isinf(d_inv_sqrt) | isnan(d_inv_sqrt)) = 0; % Guard against isolated nodes
    D_mat = spdiags(d_inv_sqrt, 0, numel(d), numel(d));
    A = D_mat * A * D_mat;
end

function [out, H] = forwardModel(m, X, A, useGraph)
    % Forward pass returning raw output logits [nLabels x nNodes] and hidden activations H
    X = X'; % Transpose to [nFeatures x nNodes] for column-wise node operations
    A = single(normalizeAdj(A));
    
    % Layer 1: Multiply by normalized adjacency if GCN, otherwise standard linear transform
    if useGraph
        Z = m.W1 * (X * A) + m.b1;
    else
        Z = m.W1 * X + m.b1;
    end
    
    H = max(Z, 0); % ReLU activation
    
    % Layer 2: Graph convolution or standard linear projection to output logits
    if useGraph
        out = m.W2 * (H * A) + m.b2;
    else
        out = m.W2 * H + m.b2;
    end
end

function [m, hist] = trainModel(m, trainG, valG, epochs, lr, useGraph)
    % Identify and print which model architecture is currently training
    if useGraph
        modelName = "GCN (Graph Convolutional Network)";
    else
        modelName = "MLP (Multi-Layer Perceptron)";
    end
    fprintf("\n==================================================\n");
    fprintf(" Training Model: %s\n", modelName);
    fprintf("==================================================\n");

    % Train model across tissue graphs using custom Adam optimization
    hist.trainF1 = zeros(epochs, 1);
    hist.valF1   = zeros(epochs, 1);
    
    % Standard Adam optimizer hyperparameters
    beta1 = 0.9; 
    beta2 = 0.999; 
    epsAdam = 1e-8;
    
    % Initialize first-moment (v) and second-moment (s) accumulators for each parameter
    v = struct(); 
    s = struct();
    names = fieldnames(m);
    for z = 1:numel(names)
        v.(names{z}) = zeros(size(m.(names{z})), "single");
        s.(names{z}) = zeros(size(m.(names{z})), "single");
    end
    
    t_step = 0; % Global update step counter for Adam bias correction
    for ep = 1:epochs
        % Shuffle tissue graph order each epoch
        order = randperm(numel(trainG));
        for q = order
            t_step = t_step + 1;
            g = trainG{q}; 
            X = single(g.X); 
            Y = single(g.Y); 
            A = sparse(g.A);
            
            % Compute binary cross-entropy loss and analytical parameter gradients
            [loss, gr] = graphGrad(m, X, Y, A, useGraph);
            
            % Apply bias-corrected Adam parameter updates
            for z = 1:numel(names)
                nm = names{z};
                v.(nm) = beta1 * v.(nm) + (1 - beta1) * gr.(nm);
                s.(nm) = beta2 * s.(nm) + (1 - beta2) * (gr.(nm).^2);
                
                v_hat = v.(nm) / (1 - beta1^t_step);
                s_hat = s.(nm) / (1 - beta2^t_step);
                
                m.(nm) = m.(nm) - lr * v_hat ./ (sqrt(s_hat) + epsAdam);
            end
        end
        
        % Record full-dataset Micro-F1 scores after each epoch
        hist.trainF1(ep) = evaluateGraphs(m, trainG, useGraph);
        hist.valF1(ep)   = evaluateGraphs(m, valG, useGraph);
        
        % Log progress every 10 epochs
        if mod(ep, 10) == 0
            fprintf("Epoch %3d | Loss: %.4f | Train F1: %.3f | Val F1: %.3f\n", ...
                ep, loss, hist.trainF1(ep), hist.valF1(ep));
        end
    end
end

function [loss, gr] = graphGrad(m, X, Y, A, useGraph)
    % Manual forward and backward pass computing multi-label BCE loss and gradients
    A = single(normalizeAdj(A));
    X = X';
    
    % --- Forward Pass ---
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
    
    O = m.W2 * HA + m.b2;                           % Raw logits [nLabels x nNodes]
    P = 1 ./ (1 + exp(-max(-30, min(30, O))));      % Clamped sigmoid probabilities
    T = Y';                                         % Ground-truth binary targets
    
    % Numerically stable binary cross-entropy loss averaged over all nodes and labels
    loss = mean(max(O, 0) - O .* T + log1p(exp(-abs(O))), "all");
    
    % --- Backward Pass (Backpropagation) ---
    dO = (P - T) / numel(T);    % Gradient w.r.t. output logits
    dW2 = dO * HA';             % Gradient w.r.t. Layer 2 weights
    db2 = sum(dO, 2);           % Gradient w.r.t. Layer 2 bias
    
    dH = m.W2' * dO;            % Backpropagate through Layer 2 linear weights
    if useGraph
        dH = dH * A;            % Backpropagate through symmetric graph convolution (A' = A)
    end
    
    dZ = dH .* (Z > 0);         % Backpropagate through ReLU activation
    dW1 = dZ * XA';             % Gradient w.r.t. Layer 1 weights
    db1 = sum(dZ, 2);           % Gradient w.r.t. Layer 1 bias
    
    gr = struct("W1", dW1, "b1", db1, "W2", dW2, "b2", db2);
end

function f = evaluateGraphs(m, G, useGraph)
    % Compute global Micro-F1 score across all tissue graphs in cell array G
    TP = 0; FP = 0; FN = 0;
    for k = 1:numel(G)
        g = G{k}; 
        [O, ~] = forwardModel(m, single(g.X), sparse(g.A), useGraph);
        P = O' > 0;             % Positive class prediction where logit > 0 (prob > 0.5)
        T = logical(g.Y);       % Ground-truth multi-label indicators
        
        % Accumulate true positives, false positives, and false negatives across all labels
        TP = TP + nnz(P & T);
        FP = FP + nnz(P & ~T);
        FN = FN + nnz(~P & T);
    end
    % Micro-averaged F1 = 2*TP / (2*TP + FP + FN)
    f = 2 * TP / max(2 * TP + FP + FN, 1);
end