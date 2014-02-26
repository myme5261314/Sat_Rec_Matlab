function [ osmMat ] = runRec(  )
%RUNREC Summary of this function goes here
%   This is the main script of this project.

params = initParams();

% x,y stands for the min-max lontitude and m,n stands for the min-max
% latitude.
[x, m] = latlon2p(params.lat_north, params.lon_west,params.z);
[y, n] = latlon2p(params.lat_south, params.lon_east,params.z);
x = floor(x);
m = floor(m);
y = ceil(y);
n = ceil(n);
% osmMat = sparse(y-x, n-m);
params.x = x;
params.y = y;
params.m = m;
params.n = n;
% params.osmMat = osmMat;

% output_args = osmMat;

[nodeMap, wayMap] = loadOSM(params);
params.nodeMap = nodeMap;
params.wayMap = wayMap;

osmMat = rasterizeOSM(params);
params.osmMat = osmMat;

disp(params);

end

