function  [x, y] = calPixelPos(mstruct,lat,lon,cdata)
width = size(cdata,1);
height = size(cdata,2);
wLat = 4;
hLon = wLat/width*height;
latArray = linspace(lat+wLat,lat-wLat,width);
lonArray = linspace(lon-hLon,lon+hLon,height);
[LAT,LON] = meshgrid(latArray,lonArray);
[x,y] = mfwdtran(mstruct,LAT,LON,[],'geosurface');
end

