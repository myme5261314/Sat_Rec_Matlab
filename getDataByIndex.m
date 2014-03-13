function [ X, Y ] = getDataByIndex( ind )
%GETDATABYINDEX Summary of this function goes here
%   This function returns the corresponding X Y vector of index ind.
XFile = 'F:/RawX.dat';
YFile = 'F:/RawY.dat';
Xsize = [64*64*3 1];
Ysize = [16*16 1];
Xtype = 'uint8';
Ytype = 'double';

X = loadData(XFile, Xsize, Xtype, ind);
Y = loadData(YFile, Ysize, Ytype, ind);
X = X';
Y = Y';


end

function result = loadData( fname, data_size, data_type, ind )
    result = [];
    fId = fopen(fname, 'r');
    if strcmp(data_type, 'uint8')
        type_byte = 1;
    elseif strcmp(data_type, 'double')
        type_byte = 8;
    end
    for i=1:length(ind)
        fseek(fId, (ind(i)-1)*prod(data_size)*type_byte, 'bof');
        oneresult = fread(fId, data_size, data_type);
        result = [result double(oneresult)];
    end
    fclose(fId);
end

