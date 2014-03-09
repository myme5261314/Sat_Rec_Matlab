function [ osmMat ] = rasterizeOSM( params )
%RASTERIZEOSM Summary of this function goes here
%   This is the function to rasterize the original OSM vector data to
%   osmMat matrix.
% osmMat = sparse(params.y-params.x+1, params.n-params.m+1);

if exist(params.cacheOSMMat, 'file')
    load(params.cacheOSMMat);
    return
end

ks = cell2mat(keys(params.wayMap));
indList = [];
count = 0;
for id=ks(1,:)
    count = count + 1;
    if mod(count,100)==0
        fprintf('%d/%d\n',count,size(ks,2))
    end
    reflist = params.wayMap(id);
    for i= 1:length(reflist)
        if i == length(reflist)
            break
        end
        indList = [indList rasterizeLine(reflist(i),reflist(i+1),params)];
    end
end
indList = unique(indList','rows')';
% [row, col] = ind2sub([params.y-params.x+1, params.n-params.m+1], indList);
% osmMat = sparse(indList(1,:), indList(2,:),1, params.y-params.x+1, params.n-params.m+1);
% Because the OSM's xy coordinate is different from the matlab's
% row/column coordinate.
osmMat = sparse(indList(2,:), indList(1,:),1, params.n-params.m+1, params.y-params.x+1);

save(params.cacheOSMMat,'osmMat');
% osmMat(indList) = 1;
% osmMat = params.osmMat;

end

function indList = rasterizeLine(ref1, ref2,params)
    if ~all((isKey(params.nodeMap,{ref1, ref2})))
        return
    end
    indList = [];
    node1 = params.nodeMap(ref1);
    node2 = params.nodeMap(ref2);
    if not (inArea(node1,params) && inArea(node2,params))
%         fprintf('Invalid node1 or node2\n');
        if ~inArea(node1,params)
%             fprintf('Invalid %f, %f\n', node1(1), node1(2))
        end
        if ~inArea(node2,params)
%             fprintf('Invalid %f, %f\n', node2(1), node2(2))
        end
        return
    end
    [x1, y1] = latlon2p(node1(1),node1(2),params.z);
    [x2, y2] = latlon2p(node2(1),node2(2),params.z);
    % transform to coorespending coordinates.
    x1 = ceil(x1 - params.x + 1);
    x2 = ceil(x2 - params.x + 1);
    y1 = y1 - params.m + 1;
    y2 = y2 - params.m + 1;
    rasterPoint = abs(x2-x1) + 1;
    rasterMatrix = [linspace(x1,x2,rasterPoint); linspace(y1,y2,rasterPoint)];
    
    rasterMatrix = ceil(rasterMatrix);
    [~, col] = find(rasterMatrix<=0);
    col = unique(col);
    rasterMatrix(:,col) = [];
    rasterMatrix = [ [rasterMatrix(1,:); rasterMatrix(2,:)] ...,
                     [rasterMatrix(1,:); rasterMatrix(2,:)+1] ...,
                     [rasterMatrix(1,:)+1; rasterMatrix(2,:)] ...,
                     [rasterMatrix(1,:)+1; rasterMatrix(2,:)+1] ...,
                   ];
    indList = rasterMatrix;
end

function result = inArea(node, params)
    lat = node(1);
    lon = node(2);
    result = (lat < params.lat_north) && (lat > params.lat_south) && ...,
                (lon < params.lon_east) && (lon > params.lon_west);
end

