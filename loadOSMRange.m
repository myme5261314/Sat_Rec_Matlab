function [ lat_south, lat_north, lon_west, lon_east ] = loadOSMRange( osmPath )
%LOADOSMRANGE Summary of this function goes here
%   This is the function to load the handling osm file's lat and lon range.

doc = xmlread(osmPath);
docRoot = doc.getDocumentElement;

bounds = docRoot.getElementsByTagName('bounds');
bounds = bounds.item(0);
lon_west = str2double(bounds.getAttribute('minlon'));
lat_south = str2double(bounds.getAttribute('minlat'));
lon_east = str2double(bounds.getAttribute('maxlon'));
lat_north = str2double(bounds.getAttribute('maxlat'));


end

