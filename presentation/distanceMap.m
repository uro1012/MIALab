clc;
clear all;
close all;

imageSize = 20;
[X, Y] = meshgrid(-imageSize:imageSize, -imageSize:imageSize);

centerX = 0;
centerY = 0;
radius = 15;
circlePixels = Y.^2 + X.^2 <= radius.^2;

f = figure();
s = subplot(2, 2, 1);
image(circlePixels) ;
colormap(s, [0 0 0;1 1 1]);
title('Binary image');
axis square;

subplot(2, 2, 2)
distMapOuter = bwdist(circlePixels);
surf(Y, X, distMapOuter)
title('Outer Euclidian distance map');
axis square

subplot(2, 2, 3)
distMapInner = bwdist(not(circlePixels));
surf(Y, X, distMapInner)
title('Inner Euclidian distance map');
axis square

subplot(2, 2, 4)
distMapSigned = distMapOuter - distMapInner;
surf(Y, X, distMapSigned)
title('Signed Euclidian distance map');
axis square