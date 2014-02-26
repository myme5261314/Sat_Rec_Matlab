function [ X Y ] = latlon2p( lat, lon, Z )
%LATLON2P Summary of this function goes here
%   This is the function to transform the lat and lon to pixel X,Y
%   coordinate at the specific level Z, X relate to lon and Y relate to
%   lat.
X = ((lon + 180) / 360) * 256 * (2^Z);
sinLat = sin(lat * pi/180);
Y = (0.5 - log((1 + sinLat) / (1 - sinLat)) / (4 * pi)) * 256 * (2^Z);
% X = int64(X);
% Y = int64(Y);

end

