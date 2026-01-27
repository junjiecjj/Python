%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function V = GetChannelNoise(N, L, mode)
% Simulate channel noise with dimensions of N times L
% mode: Noise type (Gaussian, Uniform, Laplacian, Student-T, Epsilon-Gassian-Uniform mixture, Epsilon-Gassian-Gassian mixture)

    mode = upper(mode);
    switch mode
        case 'GAUSSIAN'
            V = sqrt(1/2) * (randn(N, L) + 1j*randn(N, L));

        case 'UNIFORM'
            V = sqrt(3/2) * ((2*rand(N, L) - 1) + 1j*(2*rand(N, L) - 1));

        case 'LAPLACIAN'
            V = sqrt(1/2) * (laprnd(N, L) + 1j*laprnd(N, L));

        case 'T'
            dof = 3;
            var = dof/(dof - 2);
            V = sqrt(1/(2*var)) * (trnd(dof, N, L) + 1j*trnd(dof, N, L));

        case 'EPSILON-UNIFORM'  % Epsilon-contamination: N(0, 1) + eps*amp*Uniform(0, 1) where amp is the amplitude
            V = sqrt(1/2) * (randn(N, L) + 1j*randn(N, L));
            V_RE = real(V);
            V_IM = imag(V);

            eps = 0.1;
            No = round(L * eps);
            amp = 2.5;
            for i = 1:N
                outlier_location = randi(L, 1, No);     % Several outliers can be added onto the same location,
                                                        %    because different interference sources work independently
                V_RE(i, outlier_location) = V_RE(i, outlier_location) + amp * randi_bool(1, No);
                V_IM(i, outlier_location) = V_IM(i, outlier_location) + amp * randi_bool(1, No);
            end

            V = V_RE + 1j * V_IM;

        case 'EPSILON-GAUSSIAN'  % Epsilon-contamination: N(0, 1) + eps*N(0, Var) where Var is the variance
            V = sqrt(1/2) * (randn(N, L) + 1j*randn(N, L));
            V_RE = real(V);
            V_IM = imag(V);

            eps = 0.1;
            No = round(L * eps);
            Var = 3;
            for i = 1:N
                outlier_location = randi(L, 1, No);     % Several outliers can be added onto the same location because
                                                        %    different interference sources work independently
                V_RE(i, outlier_location) = V_RE(i, outlier_location) + sqrt(Var) * randn(1, No);
                V_IM(i, outlier_location) = V_IM(i, outlier_location) + sqrt(Var) * randn(1, No);
            end

            V = V_RE + 1j * V_IM;

        otherwise
            error('GetChannelNoise :: Error in Channel Noise Mode :: Non-existing !');
    end
end

function ret = randi_bool(N, M)
    ret = rand(N, M);
    ret(ret > 0.5) = 1;
    ret(ret < 0.5) = -1;
end