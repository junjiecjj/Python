









function H = channel(N,K)
    L = 15;
    theta = 2*pi*rand(K,L);%均匀分布在[0,2*pi)
    H = zeros(K,N);
    a = zeros(1,N);
    for x = 1:1:K % K个用户的信道
        temp = zeros(1,N);
        for l = 1:1:L
            %求 a_t (N*1)
            for n = 1:1:N
                a(n) = exp(1j *(n-1)* pi * sin(theta(x,l)));
            end
            a = a/sqrt(N);
            temp = temp + randn()*1*conj(a); %a_r = 1;
        end
        H(x,:)=sqrt(N*1/L)*temp; % M=1
    end
end
