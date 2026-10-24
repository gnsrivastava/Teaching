classdef graphConvLayer < nnet.layer.Layer & nnet.layer.Formattable
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

            % Initialize weights and bias as dlarrays (Glorot/Xavier initialization)
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
            % 1. Strip dimension labels to allow standard matrix multiplication (*)
            X_unlabeled = stripdims(X);

            numNodes = size(layer.Ahat, 1);

            % 2. Aggregate features across graph neighbors
            % (Bypasses matrix multiplication during dummy network initialization when batch size != numNodes)
            if size(X_unlabeled, 2) == numNodes
                AX = X_unlabeled * layer.Ahat; % [F_in x N] * [N x N] -> [F_in x N]
            else
                AX = X_unlabeled;
            end

            % 3. Linear transformation: [F_out x F_in] * [F_in x N] + [F_out x 1]
            Z_raw = layer.Weights * AX + layer.Bias;

            % 4. Re-format output back to "CB" (Channels x Batch) for downstream layers
            Z = dlarray(stripdims(Z_raw), "CB");
        end
    end
end