function y = bec_channel(x,e)
%Process an input vector x through a Binary Erasure Channel with
%erasure probability e.
y = x.^2; %copy the input vector, hack to convert logical to integer
erasures = (rand(1,length(x))<=e);%erasure locations
y(erasures) = -1; % erasure=-1, three valued [0,1,-1]