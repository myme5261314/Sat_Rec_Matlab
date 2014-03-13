function [ output_args ] = NNTrain( input_args )
%NNTRAIN Summary of this function goes here
%   Detailed explanation goes here
input_layer_size = 64*64*3;
hidden_layer_size = 64*64;
output_layer_size = 16*16;
lambda = 1;
load('RawXmean.mat');
load('RawXStd.mat');

%   Calculate the num of training case.
f = dir('F:/RawX.dat');
total_case = f.bytes/(64*64*3);
fprintf('The total training case is %d.\n', total_case);
case_order = randperm(total_case);

%   Randomly Initialize the NN weight Matrix.
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);

%   Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
clear initial_Theta1 initial_Theta2;

maxIter = 50;

J_record = zeros( 1, maxIter );
iter_per = 0.01;
tic;
for i=1:maxIter
    fprintf('Start of iteration %d.\n', i);
    if i/maxIter >= iter_per
        iter_per = iter_per + 0.01;
        fprintf('Iteration Process: %d/%d.\n', i, maxIter);
        toc;
        tic;
    end
    J = 0;
    grad = zeros( size(initial_nn_params) );
    case_per = 0.01;
    for j=1:total_case
        if j/total_case >= case_per
            case_per = case_per + 0.01;
            fprintf('Iteration Process: %d/%d.\n', i, maxIter);
            toc;
            tic;
        end
        [X, Y] = getDataByIndex( case_order(j) );
        X = X - RawXmean;
        X = X ./ RawXStd;
        [oneJ, onegrad] = nnCostFunction(initial_nn_params, ...
           input_layer_size, hidden_layer_size, ...
           output_layer_size, X, Y, lambda);
        J = J + oneJ;
        grad = grad + onegrad;
    end
    J_record(i) = J;
    fprintf('Start of iteration %d.\n', i);
end
toc;

plot(1:maxIter, J_record);
end




