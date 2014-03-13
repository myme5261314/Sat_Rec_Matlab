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
factor_list = factor(total_case);
batch_num = 1;
batch_threshold = 1000;
for i = size(factor_list,2):-1:1
    batch_num = batch_num * factor_list(i);
    if batch_num > batch_threshold
        break;
    end
end
batch_times = total_case/ batch_num;
fprintf('The total training case is %d.\n', total_case);
fprintf('Batch num %d with batch times %d.\n', batch_num, batch_times);
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
    for j=1:batch_times
        if j/batch_times >= case_per
            case_per = case_per + 0.01;
            fprintf('Iteration Process: %d/%d.\n', j, batch_times);
            toc;
            tic;
        end
        batch_order_list = (j-1)*batch_num+1:j*batch_num;
        [X, Y] = getDataByIndex( case_order(batch_order_list) );
%         X = X - RawXmean;
%         X = X ./ RawXStd;
        X = bsxfun(@minus, X, RawXmean);
        X = bsxfun(@rdivide, X, RawXStd);
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




