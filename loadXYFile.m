function [ XYcell ] = loadXYFile( Xfilepath, Yfilepath )
%LOADXYFILE Summary of this function goes here
%   Detailed explanation goes here
strtemplate =  'The Xfilepath is %d, The Yfilepaht is %d';
assert( all(size(Xfilepath)==size(Yfilepath)), strtemplate...
    , size(Xfilepath), size(Yfilepath) );

m = size(Xfilepath,2);
XYcell = cell(m,2);
for i=1:m
    XYcell{i,1} = imread(Xfilepath{1,i});
    XYcell{i,2} = imread(Yfilepath{1,i});
end
    

end

