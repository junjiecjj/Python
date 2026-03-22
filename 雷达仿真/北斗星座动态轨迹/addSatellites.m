function  h = addSatellites(ax,lat,lon,cdata,alpha)
width = size(cdata,1);
height = size(cdata,2);
wLat = 4;
hLon = wLat/width*height;
latArray = linspace(lat+wLat,lat-wLat,width);
lonArray = linspace(lon-hLon,lon+hLon,height);
[LAT,LON] = meshgrid(latArray,lonArray);
cdata = permute(cdata,[2,1,3]);
alpha = alpha.';
h = geoshow(ax,LAT,LON,cdata,'FaceAlpha','texturemap','AlphaData',alpha);
end
