


function [WRF,WBB] = Receiver(Wopt,NRF)
    % randomly generate FRF
    [Nt,Ns] = size(Wopt);
    WRF = [];
    for i = 1:NRF
        WRF = blkdiag(WRF, exp(sqrt(-1) * unifrnd (0,2*pi,[Nt/NRF,1])));
    end
    WRF = 1/sqrt(Nt)*WRF;

    y = [];
    while(isempty(y) || abs(y(1)-y(2))>1e-3)
        % fix FRF, optimize FBB
        WBB = pinv(WRF) * Wopt;

        y(1) = norm(Wopt-WRF*WBB,'fro')^2;

        % fix FBB, optimize FRF
        for i = 1:Nt
            m = ceil(i*NRF/Nt);
            WRF(i,m) = 1/sqrt(Nt) * exp( sqrt(-1) * angle( Wopt(i,:)*WBB(m,:)' ) );
        end
        y(2) = norm(Wopt-WRF*WBB,'fro')^2;
    end
end
