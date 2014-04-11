function [ ydata ] = yimg2data( img, windowsize, stridesize, dataIdx )
%YIMG2DATA Summary of this function goes here
%   This is the function to generate the corresponding X data entities from
%   an image of Map.

blank = (windowsize-stridesize)/2;
[rc_size] = img2matsize(size(img), windowsize, stridesize);
row_total = rc_size(1);
column_total = rc_size(2);

if nargin==3
    dataIdx = 1:row_total*column_total;
end
[row_idx, col_idx] = ind2sub([row_total, column_total], dataIdx);
idxnum = numel(dataIdx);

convertF = @(idx) blank+(idx-1)*stridesize+1;

ydata = zeros(idxnum, stridesize^2, 'uint8');

for i=1:idxnum
    ridx = row_idx(i);
    ridx = convertF(ridx);
    ridx = ridx:(ridx+stridesize-1);
    cidx = col_idx(i);
    cidx = convertF(cidx);
    cidx = cidx:(cidx+stridesize-1);
    temp = img(ridx,cidx);
    ydata(i,:) = temp(:);
end

end

