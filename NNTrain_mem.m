function [ predY ] = NNTrain_mem( params )
%NNTRAIN_MEM Summary of this function goes here
%   This is the function to train the Neural Network by loading the whole
%   X,Y data into memory.

input_layer_size = 64*64*3;
hidden_layer_size = 1;
output_layer_size = 16*16;
lambda = 1;

Xmean = mean(params.rawXmem);
Xrange = max(params.rawXmem) - min(params.rawXmem);
%   Need to use single precision to store X, because double precision X
%   exceeds the memory size.
X = bsxfun(@minus, single(params.rawXmem), single(Xmean));
X = bsxfun(@rdivide, X, single(Xrange));
assert(isa(X, 'single'));
clear Xmean Xrange;

y = params.rawYmem;

%   Randomly Initialize the NN weight Matrix.
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);

%   Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
clear initial_Theta1 initial_Theta2;

fprintf('\nTraining Neural Network... \n');
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   output_layer_size, X, y, lambda);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

save(params.cacheThetaMat, 'Theta1', 'Theta2');

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.
predY = predict(Theta1, Theta2, X);


end

