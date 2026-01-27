function d = Distance(C1, C2)
%% Calculate the distance between C1 and C2
% input
%       C1              Coordinate(x,y,z),(N×3)
%       C2              Coordinate(x,y,z),(N×3)
% output
%       d               distance between C1 and C2, N×1
%%
d = sqrt(...
    (C1(:,1)-C2(:,1)).^2+...
    (C1(:,2)-C2(:,2)).^2+...
    (C1(:,3)-C2(:,3)).^2);
end