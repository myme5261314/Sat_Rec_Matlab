function [ x, y ] = xyimgIdx2data( WindowSize, StrideSize, xyimgcell )
%XYIMGIDX2DATA Summary of this function goes here
%   Detailed explanation goes here
% xy = cell(1,2);


x = ximg2data(xyimgcell{1},...
    WindowSize, StrideSize);
y = yimg2data(xyimgcell{2},...
    WindowSize, StrideSize);


end

