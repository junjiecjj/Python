function rcsplot(P, angle, rcs, caption)
% rcsplot: plot the RCS imaging
%
%   rcsplot(P, angle, rcs)
%   the range is [0 P-1], angle is K-by-1, rcs is P-by-K

figure;
mapHot = flipud(hot);

imagesc(angle, 0:(P-1), 20*log10(abs(rcs)), [-40 0]);
colormap(mapHot);
xlabel('Angle (degree)');
ylabel('Range (round-trip subpulse delay)');
title(caption);
colorbar;
box on;
myboldify;
drawnow;