function [ osmMat ] = y2osmMat( Y ,params )
%Y2OSMMAT Summary of this function goes here
%   This is the function to tranform the Y [m n] to the osmMat form. m is
%   the data num of Y and n is the result dimension of Y. osmMat is a
%   matrix with size [params.numRows params.numColumns].
%
%   Notice: The Window stride moves with the column major order while
%   generating the Y.
%   Notice: The generated osmMat only guarantee the following equation.
%       osmMat(25:end-40,25:end-40) ==
%       params.extendMat(25:end-40,25:end-40).
[m, n] = size(Y);
% Stride = sqrt(n);
WindowSize = 64;
Stride = 16;
tF = @(l) floor((l - WindowSize)/Stride);
Xend = tF(params.numColumns);
Yend = tF(params.numRows);
assert(m == Xend*Yend);
pF = @(ind) (ind - 1)*Stride + 1 + (WindowSize-Stride)/2;
osmMat = zeros(params.numRows, params.numColumns);
for i = 1:m
    one_data = Y(i,:);
    one_data = reshape(one_data,Stride,Stride);
    Xind = floor((i-1)/Yend) + 1;
    Yind = mod((i-1), Yend) + 1;
    Xind = pF(Xind);
    Yind = pF(Yind);
    verify_mat = params.extendMat(Yind:Yind+Stride-1, Xind:Xind+Stride-1);
    verify_v = all(all(verify_mat == one_data));
%     assert(verify_v);
    osmMat(Yind:Yind+Stride-1, Xind:Xind+Stride-1) = one_data;
end
    


end

