function [p,t,filtDelay] = makeSRRCPulse(method,beta,L,span,Tsym,totalLength)
%MAKESRRCPULSE Three SRRC implementations used by the TSP reproductions.
% method = 'custom'     : exact translation of the Python srrcFunction.
% method = 'commpy'     : matches commpy.filters.rrcosfilter(totalLength,...).
% method = 'rcosdesign' : MATLAB Communications Toolbox implementation.

arguments
    method (1,:) char
    beta (1,1) double {mustBeNonnegative}
    L (1,1) double {mustBeInteger,mustBePositive}
    span (1,1) double {mustBeInteger,mustBePositive}
    Tsym (1,1) double {mustBePositive} = 1
    totalLength (1,1) double {mustBeInteger,mustBePositive} = span*L+1
end

switch lower(method)
    case 'custom'
        t = (-span*Tsym/2 : Tsym/L : span*Tsym/2).';
        p = localSRRC(t,beta,Tsym);
        % Exact behavior of the supplied Python function: NaN at t=0 is
        % replaced by 1 (rather than by the analytic SRRC limit).
        p(abs(t) < 10*eps) = 1;
        filtDelay = (numel(p)-1)/2;
        if totalLength < numel(p)
            error('totalLength is shorter than the custom SRRC pulse.');
        end
        p = [p; zeros(totalLength-numel(p),1)];

    case 'commpy'
        % scikit-commpy uses this even-length, half-sample-asymmetric grid.
        t = ((0:totalLength-1).' - totalLength/2) * Tsym/L;
        p = localSRRC(t,beta,Tsym);
        filtDelay = (numel(p)-1)/2;

    case 'rcosdesign'
        % rcosdesign returns span*L+1 samples.  Choose the requested span,
        % then centre-crop/pad to totalLength for an apples-to-apples test.
        if exist('rcosdesign','file') ~= 2
            error('rcosdesign requires Communications Toolbox.');
        end
        q = rcosdesign(beta,span,L,'sqrt').';
        if numel(q) >= totalLength
            first = floor((numel(q)-totalLength)/2)+1;
            p = q(first:first+totalLength-1);
        else
            p = [q; zeros(totalLength-numel(q),1)];
        end
        t = ((0:totalLength-1).'-(totalLength-1)/2)*Tsym/L;
        filtDelay = (numel(p)-1)/2;

    otherwise
        error('Unknown method: %s',method);
end
p = p/sqrt(sum(abs(p).^2));
end

function p = localSRRC(t,beta,Tsym)
x = t/Tsym;
num = sin(pi*x*(1-beta)) + 4*beta*x.*cos(pi*x*(1+beta));
den = pi*x.*(1-(4*beta*x).^2);
p = num./den/sqrt(Tsym);

atZero = abs(x) < 10*eps;
p(atZero) = (1 + beta*(4/pi-1))/sqrt(Tsym);
if beta > 0
    atSing = abs(abs(x)-1/(4*beta)) < 100*eps;
    p(atSing) = beta/sqrt(2*Tsym) * ...
        ((1+2/pi)*sin(pi/(4*beta)) + (1-2/pi)*cos(pi/(4*beta)));
end
end
