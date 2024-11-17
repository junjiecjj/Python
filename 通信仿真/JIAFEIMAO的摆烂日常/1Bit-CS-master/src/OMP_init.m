function [x,r,normR,residHist, errHist] = OMP( A, b, k, errFcn, opts )
% x = OMP( A, b, k )
%   uses the Orthogonal Matching Pursuit algorithm (OMP)
%   to estimate the solution to the equation
%       b = A*x     (or b = A*x + noise )
%   where there is prior information that x is sparse.
%
%   "A" may be a matrix, or it may be a cell array {Af,At}
%   where Af and At are function handles that compute the forward and transpose
%   multiplies, respectively.
%
% [x,r,normR,residHist,errHist] = OMP( A, b, k, errFcn, opts )
%   is the full version.
% Outputs:
%   'x' is the k-sparse estimate of the unknown signal
%   'r' is the residual b - A*x
%   'normR' = norm(r)
%   'residHist'     is a vector with normR from every iteration
%   'errHist'       is a vector with the outout of errFcn from every iteration
%
% Inputs:
%   'A'     is the measurement matrix
%   'b'     is the vector of observations
%   'k'     is the estimate of the sparsity (you may wish to purposefully
%              over- or under-estimate the sparsity, depending on noise)
%              N.B. k < size(A,1) is necessary, otherwise we cannot
%                   solve the internal least-squares problem uniquely.
%
%   'k' (alternative usage):
%           instead of specifying the expected sparsity, you can specify
%           the expected residual. Set 'k' to the residual. The code
%           will automatically detect this if 'k' is not an integer;
%           if the residual happens to be an integer, so that confusion could
%           arise, then specify it within a cell, like {k}.
%
%   'errFcn'    (optional; set to [] to ignore) is a function handle
%              which will be used to calculate the error; the output
%              should be a scalar
%
%   'opts'  is a structure with more options, including:
%       .printEvery = is an integer which controls how often output is printed
%       .maxiter    = maximum number of iterations
%       .slowMode   = whether to compute an estimate at every iteration
%                       This computation is slower, but it allows you to
%                       display the error at every iteration (via 'errFcn')
%
%       Note that these field names are case sensitive!
%
% If you need a faster implementation, try the very good C++ implementation
% (with mex interface to Matlab) in the "SPAMS" toolbox, available at:
%   http://www.di.ens.fr/willow/SPAMS/
% The code in SPAMS is precompiled for most platforms, so it is easy to install.
% SPAMS uses Cholesky decompositions and uses a slightly different
%   updating rule to select the next atom.
%
% Stephen Becker, Aug 1 2011.  srbecker@alumni.caltech.edu
% Updated Dec 12 2012, fixing bug for complex data, thanks to Noam Wagner.
%   See also CoSaMP, test_OMP_and_CoSaMP



if nargin < 5, opts = []; end
if ~isempty(opts) && ~isstruct(opts)
    error('"opts" must be a structure');
end

function out = setOpts( field, default )
    if ~isfield( opts, field )
        opts.(field)    = default;
    end
    out = opts.(field);
end

slowMode    = setOpts( 'slowMode', false );
printEvery  = setOpts( 'printEvery', 50 );

% What stopping criteria to use? either a fixed # of iterations,
%   or a desired size of residual:
target_resid    = -Inf;
if iscell(k)
    target_resid = k{1};
    k   = size(b,1);
elseif k ~= round(k)
    target_resid = k;
    k   = size(b,1);
end
% (the residual is always guaranteed to decrease)
if target_resid == 0 
    if printEvery > 0 && printEvery < Inf
        disp('Warning: target_resid set to 0. This is difficult numerically: changing to 1e-12 instead');
    end
    target_resid    = 1e-12;
end
    

if nargin < 4
    errFcn = [];   
elseif ~isempty(errFcn) && ~isa(errFcn,'function_handle')
    error('errFcn input must be a function handle (or leave the input empty)');
end

if iscell(A)
    LARGESCALE  = true;
    Af  = A{1};
    At  = A{2};     % we don't really need this...
else
    LARGESCALE  = false;
    Af  = @(x) A*x;
    At  = @(x) A'*x;
end

% -- Intitialize --
% start at x = 0, so r = b - A*x = b
r           = b-A*opts.X0;
normR       = norm(r);
Ar          = At(r);
N           = size(Ar,1);       % number of atoms
M           = size(r,1);        % size of atoms
if k > M
    %error('K cannot be larger than the dimension of the atoms');
end
unitVector  = zeros(N,1);
x           = opts.X0;

indx_set    = zeros(k,1);
indx_set_sorted     = zeros(k,1);
A_T         = zeros(M,k);
A_T_nonorth = zeros(M,k);
residHist   = zeros(k,1);
errHist     = zeros(k,1);

if ~isempty(errFcn) && slowMode
    %fprintf('Iter,  Resid,   Error\n');
else
    %fprintf('Iter,  Resid\n');
end

for kk = 1:k
    
    % -- Step 1: find new index and atom to add
    [dummy,ind_new]     = max(abs(Ar));
    % Check if this index is already in
%     if ismember( ind_new, indx_set_sorted(1:kk-1) )
%         disp('Shouldn''t happen... entering debug');
%         keyboard
%     end
    
    
    indx_set(kk)    = ind_new;
    indx_set_sorted(1:kk)   = sort( indx_set(1:kk) );
    
    if LARGESCALE
        unitVector(ind_new)     = 1;
        atom_new                = Af( unitVector );
        unitVector(ind_new)     = 0;
    else
        atom_new    = A(:,ind_new);
    end
    
    A_T_nonorth(:,kk)   = atom_new;     % before orthogonalizing and such
    
    
    
    % -- Step 2: update residual
    
    if slowMode
        % The straightforward way:
        x_T = A_T_nonorth(:,1:kk)\b;
        
        % or, use QR decomposition:
%         if kk < 10
% %             [Q,R] = qr( A_T_nonorth(:,1:kk), 0 );
%             [Q,R] = qr( A_T_nonorth(:,1:kk)); % need full "Q" matrix to use "qrinsert"
%             %  For this reason, "qrinsert" is not efficient
%         else
%             % from now on, we use the old QR to update the new one
%             [Q,R] = qrinsert( Q, R, kk, atom_new );
%         end
%         x_T = R\(R'\(A_T_nonorth(:,1:kk)'*b));
        
        
        x( indx_set(1:kk) )   = x_T;
        r   = b - A_T_nonorth(:,1:kk)*x_T;
    else
    
        % First, orthogonalize 'atom_new' against all previous atoms
        % We use MGS
        for j = 1:(kk-1)
%             atom_new    = atom_new - (atom_new'*A_T(:,j))*A_T(:,j);
            % Thanks to Noam Wagner for spotting this bug. The above line
            % is wrong when the data is complex. Use this:
            atom_new    = atom_new - (A_T(:,j)'*atom_new)*A_T(:,j);
        end
        % Second, normalize:
        atom_new        = atom_new/norm(atom_new);
        A_T(:,kk)       = atom_new;
        % Third, solve least-squares problem (which is now very easy
        %   since A_T(:,1:kk) is orthogonal )
        x_T     = A_T(:,1:kk)'*b;
        x( indx_set(1:kk) )   = x_T;      % note: indx_set is guaranteed to never shrink
        % Fourth, update residual:
        %     r       = b - Af(x); % wrong!
        r       = b - A_T(:,1:kk)*x_T;
        
        % N.B. This err is unreliable, since this "x" is not the same
        %   (since it relies on A_T, which is the orthogonalized version).
    end
    
    
    normR   = norm(r);
    % -- Print some info --
    PRINT   = ( ~mod( kk, printEvery ) || kk == k );
    if printEvery > 0 && printEvery < Inf && (normR < target_resid )
        % this is our final iteration, so display info
        PRINT = true;
    end

    if ~isempty(errFcn) && slowMode
        er  = errFcn(x);
    %    if PRINT, fprintf('%4d, %.2e, %.2e\n', kk, normR, er ); end
        errHist(kk)     = er;
    else
    %    if PRINT, fprintf('%4d, %.2e\n', kk, normR ); end
        % (if not in SlowMode, the error is unreliable )
    end
    residHist(kk)   = normR;
    

    
    if normR < target_resid
        if PRINT
            fprintf('Residual reached desired size (%.2e < %.2e)\n', normR, target_resid );
        end
        break;
    end
    
    if kk < k
        Ar  = At(r); % prepare for next round
    end
    
end
if (target_resid) && ( normR >= target_resid )
    fprintf('Warning: did not reach target size of residual\n');
end


if ~slowMode  % (in slowMode, we already have this info)
 % For the last iteration, we need to do this without orthogonalizing A
 % so that the x coefficients match what is expected.
 x_T = A_T_nonorth(:,1:kk)\b;
 x( indx_set(1:kk) )   = x_T;
end
r       = b - A_T_nonorth(1:kk)*x_T;
normR   = norm(r);

end % end of main function
