function geoTable = calSatPos(ecefTable)
bdcs = oblateSpheroid;
bdcs.SemimajorAxis = 6378137;
bdcs.InverseFlattening = 298.257222101;
f = @(z) ecef2geodetic(bdcs,z(1),z(2),z(3));
geoTable = rowfun(f,ecefTable,'InputVariables','pos','NumOutputs',3);
end