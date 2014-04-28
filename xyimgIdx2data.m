function [ x, y ] = xyimgIdx2data( data_per_img, WindowSize, StrideSize, xyimgcell, dataIdx )
%XYIMGIDX2DATA Summary of this function goes here
%   Detailed explanation goes here
% xy = cell(1,2);

if nargin==4
    dataIdx = 1:data_per_img;
end

x = ximg2data(xyimgcell{1},...
    WindowSize, StrideSize, dataIdx);
y = yimg2data(xyimgcell{2},...
    WindowSize, StrideSize, dataIdx);


end

