function result = partExec(mat, batchsize, exeFun)
% This is the generalize version for part-wise calculate mean, std, cov and
% some operation like that part result + part result / total num case.
m = size(mat, 1);
for i = 1:batchsize:m
    if i==m-1
        continue;
    end
    temp = mat(i:i+batchsize-1, :);
    temp = exeFun(double(temp));
    if i==1
        result = temp;
    else
        result = result + temp;
    end
end
r = mod(m, batchsize);
if r ~= 0
    temp = mat(end-r+1:end, :);
    temp = exeFun(double(temp));
    result = result + temp;
end
% result = result/m;

end
