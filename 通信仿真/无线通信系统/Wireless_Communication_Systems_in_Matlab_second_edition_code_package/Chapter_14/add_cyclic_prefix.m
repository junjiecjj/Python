function s = add_cyclic_prefix(x,Ncp)
%function to add cyclic prefix to the generated OFDM symbol x that
%is generated at the output of the IDFT block
% x - ofdm symbol without CP (output of IDFT block)
% Ncp-num. of samples at x's end that will copied to its beginning
% s - returns the cyclic prefixed OFDM symbol
s = [x(end-Ncp+1:end) x]; %Cyclic prefixed OFDM symbol
end