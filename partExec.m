function result = partExec(mat, batchsize, exeFun, gpuEnable)
% This is the generalize version for part-wise calculate mean, std, cov and
% some operation like that part result + part result / total num case.
m = size(mat, 1);
idx = 1:batchsize:m;
if idx(end) ~= m
    idx = [idx m];
end
for i = 1:size(idx,2)-1
    if i ~= size(idx,2)-1
        temp = mat(idx(i):idx(i+1)-1, :);
    else
        temp = mat(idx(i):idx(i+1), :);
    end
    if nargin==4 && strcmp(gpuEnable, 'gpu')
        temp = gpuArray(temp);
    end
    temp = exeFun(double(temp));
    if i==1
        result = temp;
    else
        result = result + temp;
    end
end
% result = result/m;

end
