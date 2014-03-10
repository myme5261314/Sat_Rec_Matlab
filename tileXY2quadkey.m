function [ quadkey ] = tileXY2quadkey( X, Y, Z )
%TILEXY2QUADKEY Summary of this function goes here
%   This is the function to transform the tile X,Y to the quadkey at level
%   Z.
if nargin<3
    X = 0;
    Y = 0;
    Z = 19;
end
quadkey = '';
for i=Z:-1:1
    digit = 0;
    mask = bitshift(1, i-1);
    if(bitand(X, mask)) ~= 0
        digit = digit + 1;
    end
    if(bitand(Y, mask)) ~= 0
        digit = digit + 2;
    end
%     This line of the code runs too slow than the below one
%     quadkey1 = strcat(quadkey, num2str(digit));
    quadkey = sprintf('%s%d',quadkey, digit);
end


end

