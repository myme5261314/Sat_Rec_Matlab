function [ predyimgcell ] = predy2img( data_per_img, datasize_per_img, predy )
%PREDY2IMG Summary of this function goes here
%   Detailed explanation goes here
assert( mod(size(predy,1), data_per_img)==0 );
m = size(predy,1)/data_per_img;
predyimgcell = cell(m,1);
% thresholdlist = unique(predy(:));
for i=1:m
    predyimgcell{i,1} = zeros(datasize_per_img*16, 'single');
    yidx = ( (i-1)*data_per_img+1 ):i*data_per_img;
    ydata = predy(yidx,:);
    [rows, cols] = ind2sub(datasize_per_img, 1:size(ydata,1));
    rows = (rows-1) * 16 + 1;
    cols = (cols-1) * 16 + 1;
    for l=1:size(rows,2)
        ridx_s = rows(l);
        ridx_e = ridx_s + 15;
        cidx_s = cols(l);
        cidx_e = cidx_s + 15;
        predyimgcell{i,1}(ridx_s:ridx_e, cidx_s:cidx_e) = reshape(ydata(l,:), 16, 16);
    end
end
        

end
