function [ nn ] = nn_setup( input_layer_size, hidden_layer_size, output_layer_size, opts )
%NN_SETUP Summary of this function goes here
%   Detailed explanation goes here
% nn.Theta1 = w';
nn.Theta1 = single(randInitializeWeights(input_layer_size, hidden_layer_size));
nn.vTheta1 = zeros(size(nn.Theta1), 'single');
nn.Theta2 = single(randInitializeWeights(hidden_layer_size, output_layer_size));
nn.vTheta2 = zeros(size(nn.Theta2), 'single');

nn.alpha = opts.alpha;
nn.momentum = opts.momentum;
nn.L2 = opts.L2;

end

