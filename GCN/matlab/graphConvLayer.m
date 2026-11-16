classdef graphConvLayer < nnet.layer.Layer & nnet.layer.Formattable
    % GRAPHCONVLAYER Custom Graph Convolutional Layer for Deep Learning Toolbox.
    % Computes spatial aggregation (neighbor message passing) followed by a 
    % learnable linear feature transformation.

    properties
        Ahat
        InputSize
        OutputSize
    end

    properties (Learnable)
        Weights
        Bias
    end

    methods
        function layer = graphConvLayer(inputSize, outputSize, Ahat, name)
            layer.Name = name;
            layer.Description = "Graph Convolution Layer";
            layer.InputSize = inputSize;
            layer.OutputSize = outputSize;

            % Store Ahat as a dense single-precision array
            layer.Ahat = single(full(Ahat));

            % Initialize weights and bias (Glorot/Xavier uniform initialization)
            lim = sqrt(6 / (inputSize + outputSize));
            layer.Weights = dlarray(single((2 * rand(outputSize, inputSize) - 1) * lim));
            layer.Bias = dlarray(zeros(outputSize, 1, "single"));
        end

        function Z = predict(layer, X)
            Z = layer.forwardPass(X);
        end

        function Z = forward(layer, X)
            Z = layer.forwardPass(X);
        end
    end

    methods (Access = private)
        function Z = forwardPass(layer, X)
            % Strip dimension labels to allow standard matrix multiplication (*)
            X_unlabeled = stripdims(X);
            numNodes = size(layer.Ahat, 1);

            % Graph message passing (neighbor aggregation)
            if size(X_unlabeled, 2) == numNodes
                AX = X_unlabeled * layer.Ahat; % [F_in x N] * [N x N] -> [F_in x N]
            else
                AX = X_unlabeled; % Fallback for dlnetwork initialization pass
            end

            % Linear transformation: W * AX + b
            Z_raw = layer.Weights * AX + layer.Bias;

            % Re-format tensor for downstream Deep Learning Toolbox layers
            Z = dlarray(stripdims(Z_raw), "CB");
        end
    end
end