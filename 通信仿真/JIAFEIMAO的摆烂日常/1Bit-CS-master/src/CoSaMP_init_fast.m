function [x,r,normR] = CoSaMP_init_fast( A, b, k, opts )
% modified to add initialization and use fast implementation, by S.H.

% x = CoSaMP( A, b, k )
%   uses the Compressive Sampling Matched Pursuit (CoSaMP) algorithm
%   (see Needell and Tropp's 2008 paper http://arxiv.org/abs/0803.2392 )
%   to estimate the solution to the equation
%       b = A*x     (or b = A*x + noise )
%   where there is prior information that x is sparse.
%
%   "A" may be a matrix, or it may be a cell array {Af,At}
%   where Af and At are function handles that compute the forward and transpose
%   multiplies, respectively. If the function handles are provided,
%   the the least-squares step is performed using LSQR (use could also use
%   CG on the normal equations, or other special variants).
%
% [x,r,normR,residHist,errHist] = CoSaMP( A, b, k, errFcn, opts )
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
%   'errFcn'    (optional; set to [] to ignore) is a function handle
%              which will be used to calculate the error; the output
%              should be a scalar
%   'opts'  is a structure with more options, including:
%       .printEvery = is an integer which controls how often output is printed
%       .maxiter    = maximum number of iterations
%       .normTol    = desired size of norm(residual). This is also
%                       used to detect convergence when the residual
%                       has stopped decreasing in norm
%       .LSQR_tol   = when "A" is a set of function handles, this controls
%                       the tolerance in the iterative solver. For compatibility
%                       with older versions, the fieldname "cg_tol" is also OK.
%       .LSQR_maxit = maximum number of steps in the iterative solver. For compatibility
%                       with older versions, the fieldname "cg_maxit" is also OK.
%                       N.B. "CG" stands for conjugate gradient, but this code
%                            actually uses the LSQR solver.
%       .HSS        = if true, use the variant of CoSaMP that is similar to
%                       HHS Pursuit (see appendix A.2 of Needell/Tropp paper). Recommended.
%       .two_solves = if true, uses the variant of CoSaMP that re-solves
%                       on the support of size 'k' at every iteration
%                       (see appendix). This can be used with or without HSS variant.
%       .addK       = the number of new entries to add each time. By default
%                       this is 2*k (this was what is used in the paper).
%                       If you experience divergence, try reducing this.
%                       We recommend trying 1*k for most problems.
%       .support_tol = when adding the (addK) atoms with the largest
%                       correlations, the CoSaMP method specifies that you do
%                       not add the atoms if the correlation is exactly zero.
%                       In practice, it is better to not add the atoms of their
%                       correlation is nearly zero. "support_tol" controls
%                       what this 'nearly zero' number is, e.g. 1e-10.
%
%       Note that these field names are case sensitive!
%
%
% Stephen Becker, Aug 1 2011  srbecker@alumni.caltech.edu
%   Updated Dec 12 2012
%   See also OMP, test_OMP_and_CoSaMP

function out = setOpts( field, default )
    if ~isfield( opts, field )
        opts.(field)    = default;
    end
    out = opts.(field);
end

maxiter     = setOpts( 'maxiter', 1000 );
normTol     = setOpts( 'normTol', 1e-10 );

cg_tol      = setOpts( 'cg_tol', 1e-6 );
cg_maxit    = setOpts( 'cg_maxit', 100 );


% Allow some synonyms
addK        = min(round(2*k), size(A,2));
support_tol = setOpts( 'support_tol', 1e-10 );

Af  = @(x) A*x;
At  = @(x) A'*x;

% -- Intitialize --
% start at x = 0, so r = b - A*x = b
r           = b-A*opts.X0;
Ar          = At(r);
N           = size(Ar,1);       % number of atoms
M           = size(r,1);        % size of atoms
%if k > M/3
%    error('K cannot be larger than the dimension of the atoms');
%end
x           = opts.X0;
idx_tmp     = 1:length(x);
ind_k       = idx_tmp(x~=0);

r_older     = [];
r_old       = [];

for kk = 1:maxiter
    
    % -- Step 1: find new index and atom to add
    y_sort      = sort( abs(Ar),'descend');
    cutoff      = y_sort(addK); % addK is typically 2*k
    cutoff      = max( cutoff, support_tol );
    ind_new     = find( abs(Ar) >= cutoff );
    

    % -- Merge:
    T    = union( ind_new, ind_k );
    RHS     = b;
    x_warmstart     = x(T);
    % -- solve for x on the suppor set "T"
    %x_T = A(:,T)\RHS;   % more efficient; equivalent to pinv when underdetermined.

    % use lsqr to solve large scale problems
    [x_T,flag,relres,CGiter] = lsqr(A(:,T), RHS, cg_tol, cg_maxit, [], [], x_warmstart);
    % Standard CoSaMP
    % Note: this is implemented *slightly* more efficiently
    %   that the HSS variation
    
    % Prune x to keep only "k" entries:
    cutoff  = findCutoff(x_T, k);
    Tk      = find( abs(x_T) >= cutoff );
    % This is assuming there are no ties. If there are,
    %    from a practical standpoint, it probably doesn't
    %    matter much what you do. So out of laziness, we don't worry about it.
    ind_k   = T(Tk);
    x       = 0*x;
    x( ind_k ) = x_T( Tk );
        
    % Update x and r
    r_older = r_old;
    r_old   = r;
    % don't do a full matrix-vector multiply, just use necessary columns
    r   = b - A(:,ind_k)*x_T( Tk );
    
    % -- Print some info --
    normR   = norm(r);

    %if (kk>=3)
    %    fprintf('Iteration %d\t%f\t%f\t%f\n', kk, normR, norm( r - r_old ), norm(r-r_older));
    %else
    %    fprintf('Iteration %d\t%f\t%f\n', kk, normR, norm( r - r_old ));
    %end

    STOP    = false;
    if normR < normTol || norm( r - r_old ) < normTol
        STOP    = true;
    end

    % check for oscillation
    if (kk>=3)
        if norm( r - r_older ) < normTol
            STOP    = true;
        end
    end
    
    if STOP
        disp('Reached stopping criteria');
        break;
    end

    if kk < maxiter
        Ar  = At(r); % prepare for next round
    end
    
end

end % -- end of main function

function tau = findCutoff( x, k )
% finds the appropriate cutoff such that after hard-thresholding,
% "x" will be k-sparse
x   = sort( abs(x),'descend');
if k > length(x)
    tau = x(end)*.999;
else
    tau  = x(k);
end
end
