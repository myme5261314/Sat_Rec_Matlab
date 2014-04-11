function [ Xfilepath, Yfilepath ] = getDataSetFilePath( rootFloder, splitSetFloder, xFloder, yFloder )
%GETDATASETFILEPATH Summary of this function goes here
%   This is the function to return the the splitSetFloder part(i.e. Train|
%   Valid| Test) under the rootFloder.
%   @return Xfilepath size(1, filenum) cell with every element a whole
%   path.
%   @return Yfilepath size(1, filenum) same with Yfilepath.
Xfile = dir( fullfile(rootFloder, splitSetFloder, xFloder , '*.tiff' ) );
Xfile = struct2cell(Xfile);
Xfilepath = fullfile(rootFloder, splitSetFloder, xFloder, Xfile(1,:));
Yfile = dir( fullfile(rootFloder, splitSetFloder, yFloder , '*.tif' ) );
Yfile = struct2cell(Yfile);
Yfilepath = fullfile(rootFloder, splitSetFloder, yFloder, Yfile(1,:));

strtemplate = 'The Xfilepath is %d, The Yfilepaht is %d';
assert( all(size(Xfilepath)==size(Yfilepath)), strtemplate...
    , size(Xfilepath), size(Yfilepath) );


end

