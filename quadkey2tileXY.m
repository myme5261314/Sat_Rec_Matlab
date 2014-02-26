function [ X, Y, Z ] = quadkey2tileXY( quadkey )
%QUADKEY2TILEXY Summary of this function goes here
%   This is the function to transform the quadkey to tile X,Y at level Z.

X = 0;
Y = 0;
Z = length(quadkey);
for i=[0:Z-1]
    mask = bitshift(1, Z - i - 1);
    temp = quadkey(i+1);
    if temp == '0'
        continue
    end

    if temp == '1'
        X = bitor(X, mask);
        continue
    end

    if temp == '2'
        Y = bitor(Y, mask);
        continue
    end

    if temp == '3'
        X = bitor(X, mask);
        Y = bitor(Y, mask);
        continue
    end


end

