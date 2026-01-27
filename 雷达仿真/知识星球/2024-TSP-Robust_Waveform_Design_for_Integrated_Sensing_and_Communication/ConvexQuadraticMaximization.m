%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function h = ConvexQuadraticMaximization(C, s, hBar, theta)
%% Implementation of Proposition 5
% Solution to
%       max     ||C*h - s||^2_2               # Vector 2-Norm
%       s.t.    ||h - hBar||_2 <= theta       # We use 2-Norm here so that Proposition 7 applicable
% where C is a real-valued matrix; h, s, hBar are real-valued vectors; theta is a scalar

    h = hBar + 1e-3*theta*randn(size(hBar));
    % Verify the initialization condition
    assert(norm(h - hBar) <= theta);              % OK
    assert(norm(h - (C'*C)^-1*C'*s) >= 1e-6);     % OK

    % funValue = norm_square(C*h-s, 2);           % For checking convergence only

    MaxIter = 100;      % Converges only with several steps; this algorithm is very fast due to closed-form solutions.
    while true
        % Proposition 6 to Solve (42)
        y = GetY(C, h, s);

        % Proposition 7 to solve (43)
        % GetH
        h = theta*((C'*C)*y - C'*s)/norm((C'*C)*y - C'*s) + hBar;
        % Verify the solution: Does h maximize the objective while satisfying the constraint?
        % assert((y'*(C'*C) - s'*C)*(h - hBar) >= 0);   % OK

        if (C'*C*y - C'*s)'*(h - y) <= 0 || MaxIter < 0     % The "MaxIter < 0" never triggered on my hand. I wrote this just for protection; recall software security.
            break;
        end

        MaxIter = MaxIter - 1;

        % % For checking convergence only
        % funValue = [funValue, norm_square(C*h-s, 2)];
        % plot(1:length(funValue), funValue, 'r-o', 'linewidth', 2, 'markersize', 10)
    end
end

function y = GetY(C, h, s)
    MethodMode = 1;
    switch MethodMode
        case 1
            M = chol(C'*C);     % such that M'*M = C'*C;
            gamma = sqrt((norm(C*h-s))^2 + s'*C*(C'*C)^-1*C'*s - s'*s);

            % Verify the correctness of the definition of gamma
            % assert(abs(norm(M*h-(M^-1)'*C'*s) - gamma) < 1e-6);      % OK

            num = (M^-1)'*((C'*C)*h - C'*s);     % numerator
            y = M^-1*(gamma * num/norm(num) + (M^-1)'*C'*s);

            % Verify the solution: Does y maximize the objective while satisfying the constraint?
            % assert((h'*(C'*C) - s'*C)*(y - h) >= -1e-6);          % OK    % Some numerical stability issues here. So I used "-1e-6", not strictly as "0".
            % assert(abs(norm(C*y - s) - norm(C*h - s)) < 1e-6);    % OK

            % A numerical protection; in principle, it should never happend; but in practice, this happens when numerical stability issues.
            % Note that "(h'*(C'*C) - s'*C)*(y - h) >= -1e-6" is always true.
            if (h'*(C'*C) - s'*C)*(y - h) < 0
                y = h;
            end

        case 2
            y = h;  % A good practical trick to speed up            % It makes no much difference compared to "case 1"

        otherwise
            error('ConvexQuadraticMaximization :: GetY :: Error in MethodMode :: Non-exist!');
    end
end
