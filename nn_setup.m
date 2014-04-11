function [ nn ] = nn_setup( params, w, opts )
%NN_SETUP Summary of this function goes here
%   Detailed explanation goes here
nn.Theta1 = w';
nn.vTheta1 = zeros(size(nn.Theta1), 'single');
nn.Theta2 = single(randInitializeWeights(4096*3, 256));
nn.vTheta2 = zeros(size(nn.Theta2), 'single');

nn.alpha = opts.alpha;
nn.momentum = opts.momentum;
nn.L2 = opts.L2;

end

