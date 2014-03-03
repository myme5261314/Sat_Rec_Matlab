function [ nodeMap, wayMap ] = loadOSM( params )
%LOADOSM Summary of this function goes here
%   This is the function to load the osm file in params.

if exist(params.cacheOSM, 'file')
    load(params.cacheOSM);
    return
end

osm = xmlread(params.osmFile);
osmRoot = osm.getDocumentElement;
nodeMap = containers.Map('KeyType','int64','ValueType','any');
nodes = osmRoot.getElementsByTagName('node');
for i=0:nodes.getLength()-1
    node_id = str2double(nodes.item(i).getAttribute('id'));
    node_id = int64(node_id);
    node_lat = str2double(nodes.item(i).getAttribute('lat'));
    node_lon = str2double(nodes.item(i).getAttribute('lon'));
    nodeMap(node_id) = [node_lat node_lon];
end

wayMap = containers.Map('KeyType','int64','ValueType','any');
ways = osmRoot.getElementsByTagName('way');
for i=0:ways.getLength()-1
    way_id = str2double(ways.item(i).getAttribute('id'));
    way_id = int64(way_id);
    nodelist = ways.item(i).getChildNodes();
    reflist = [];
    for j=0:nodelist.getLength()-1
        if strcmp(nodelist.item(j).getNodeName(),'nd')
            reflist = [reflist int64(str2double(nodelist.item(j).getAttribute('ref')))];
        end
    end
    wayMap(way_id) = reflist;
end

save(params.cacheOSM, 'nodeMap', 'wayMap');


end

