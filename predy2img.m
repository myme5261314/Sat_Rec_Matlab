function [ predyimgcell, thresholdlist ] = predy2img( params, predy )
%PREDY2IMG Summary of this function goes here
%   Detailed explanation goes here
assert( mod(size(predy,1), params.data_per_img)==0 );
m = size(predy,1)/params.data_per_img;
predyimgcell = cell(m,1);
thresholdlist = unique(predy(:));
for i=1:m
    predyimgcell{i,1} = zeros(params.datasize_per_img*params.StrideSize, 'single');
    yidx = ( (i-1)*params.data_per_img+1 ):i*params.data_per_img;
    ydata = predy(yidx,:);
    [rows, cols] = ind2sub(params.datasize_per_img, 1:size(ydata,1));
    rows = (rows-1) * params.StrideSize + 1;
    cols = (cols-1) * params.StrideSize + 1;
    for l=1:size(rows,2)
        ridx_s = rows(l);
        ridx_e = ridx_s + params.StrideSize - 1;
        cidx_s = cols(l);
        cidx_e = cidx_s + params.StrideSize - 1;
        predyimgcell{i,1}(ridx_s:ridx_e, cidx_s:cidx_e) = reshape(ydata(l,:), params.StrideSize, params.StrideSize);
    end
end
        

end