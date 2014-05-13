function [ xdata ] = ximg2data( img, windowsize, stridesize )
%XIMG2DATA Summary of this function goes here
%   This is the function to generate the corresponding X data entities from
%   an image of Arieal.

[rc_size] = img2matsize(size(img), windowsize, stridesize);
row_total = rc_size(1);
column_total = rc_size(2);
if nargin==3
    dataIdx = 1:row_total*column_total;
end
[row_idx, col_idx] = ind2sub([row_total, column_total], dataIdx);
idxnum = numel(dataIdx);

convertF = @(idx) (idx-1)*stridesize+1;

xdata = zeros(idxnum, windowsize*windowsize*3, 'uint8');

for i=1:idxnum
    ridx = row_idx(i);
    ridx = convertF(ridx);
    ridx = ridx:(ridx+windowsize-1);
    cidx = col_idx(i);
    cidx = convertF(cidx);
    cidx = cidx:(cidx+windowsize-1);
    temp = img(ridx,cidx,:);
    xdata(i,:) = temp(:)';
end

end

