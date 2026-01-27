%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function [robust_cost, Xr, Hr] = GetRobustSolution_Method1(HBar, S, F, theta)
    %% Implementation of Method 1

    %% Initialization
    [UBar, ~, VBar] = svd(F'*HBar'*S);
    [K, N] = size(HBar);
    [~, L] = size(S);
    if N >= L
        I = [eye(L, L); zeros(N-L, L)];
    else
        I = [eye(N, N), zeros(N, L-N)];
    end
    ABar = UBar*I*VBar';

    % From (38) to (41) using Lemma 3
    C__ = sqrt(L)*kron((F*ABar).', eye(K));
    Gamma = real(C__); 
    Theta = imag(C__);
    C = [Gamma -Theta; Theta Gamma];
    s    = VectorizeComplex(S);
    hBar = VectorizeComplex(HBar);

    %% Main Algorithm (Method 1)
    hStar = ConvexQuadraticMaximization(C, s, hBar, theta);     % Proposition 5
    HStar = ComplexMatrixize(hStar, K, N);
    [UStar, VStar] = RemedyProblem(HStar, S, F, ABar);          % Proposition 8
    AStar = UStar*I*VStar';
   
    %% Solution from the solving process
    Xr = sqrt(L)*F*AStar;
    Hr = HStar;
    robust_cost = (norm(Hr*Xr - S, 'fro'))^2;
end

function [UStar, VStar] = RemedyProblem(HStar, S, F, ABar)
    %% Implementation of Proposition 8
    alpha = 10000;     % let alpha = 10000, the sqrt root is 100;
    
    %% Initialization (Remark 5)
    [U, Sigma, V] = svd(F'*HStar'*S);
    [~, N] = size(HStar);
    [~, L] = size(S);
    if N >= L
        I = [eye(L, L); zeros(N-L, L)];
    else
        I = [eye(N, N), zeros(N, L-N)];
    end
    
    %% Algorithm Parameters
    round = 5000;               % the maximum iteration to await convergence
    cvg_tolerance = 1e-6;       % the numerical threshold to claim numerical convergence (cvg)
    cvg_status = zeros(3, 1);   % to record whether the three componnets (i.e., U, Sigma, V) converge
    accuracy_tolerance = 1e-3;  % to see whether the numerical errors are under tolerance level when the iteration terminated
    
    UOld = randn(size(U));
    SigmaOld = randn(size(Sigma));
    VOld = randn(size(V));

    % For checking convergence only: Very fast, converges in several steps only due to closed-form solutions
    % funValue = alpha * norm_square(U*Sigma*V' - F'*HStar'*S, 'fro') + norm_square(U*I*V' - ABar, 'fro');
    
    %% Solve
    while true
        %% Solve U
        U = OPP([I*V', sqrt(alpha)*Sigma*V'], [ABar, sqrt(alpha)*F'*HStar'*S]);
        if norm(U-UOld) < cvg_tolerance
            cvg_status(1) = 1;
        else
            cvg_status(1) = 0;
        end
    
        %% Solve V
        V = OPP([I'*U', sqrt(alpha)*Sigma'*U'], [ABar', sqrt(alpha)*S'*HStar*F]);
        if norm(V-VOld)  < cvg_tolerance
            cvg_status(2) = 1;
        else
            cvg_status(2) = 0;
        end
    
        %% Solve Sigma
        Sigma = Projection(U'*F'*HStar'*S*V, I);    % Project the former onto the subspace of the latter

        if norm(Sigma-SigmaOld) < cvg_tolerance
            cvg_status(3) = 1;
        else
            cvg_status(3) = 0;
        end

        % For checking convergence only
        % funValue = [funValue, alpha * norm_square(U*Sigma*V' - F'*HStar'*S, 'fro') + norm_square(U*I*V' - ABar, 'fro')];
        % plot(1:length(funValue), funValue, 'r-o', 'linewidth', 2, 'markersize', 10);
    
        %% Termination Control
        round = round - 1;
        if sum(cvg_status) >= 2 || round <= 0
            try
                %The algorithm is allowed to use a small "round" for quick ...
                %  termination. But the minimum numerical accuracy level must be reached.
                %  Otherwise, the solutions returned are not acceptable (recall the SVD operation).
                assert(norm(U*Sigma*V' - F'*HStar'*S) < accuracy_tolerance);
                assert(norm(V'*V - eye(L))            < accuracy_tolerance);
                assert(norm(U'*U - eye(N))            < accuracy_tolerance);
            catch
                warning('GetRobustSolution_Method1 :: Error in Accuracy Assertion :: Improve the iteration budget for better convergence.');
            end
    
            break;
        end
    
        %% Record values in this round
        UOld = U;
        SigmaOld = Sigma;
        VOld = V;
    end

    UStar = U;
    VStar = V;
end

function Sigma = Projection(Mat, I)
% Remark 6
    ProjectionMode = 1;
    switch ProjectionMode                 % Believe it or not: no much difference between the two methods
        case 1
            [~, Sigma, ~] = svd(Mat);
        case 2
            Sigma = Mat .* I;             % Projection: Set Zeroes in Non-diagnals
            Sigma = real(Sigma);          % Projection: Only Keep Real Components in Diagnal
            Sigma(Sigma < 0) = 0;         % Projection: Set Negatives to Zeroes
        otherwise
            error('GetRobustSolution_Method1 :: Projection :: Error in Mode :: Non-exist!');
    end
end
