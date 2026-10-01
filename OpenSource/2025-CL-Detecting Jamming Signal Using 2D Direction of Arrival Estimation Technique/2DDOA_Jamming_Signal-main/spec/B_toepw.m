function T=B_toepw(u,M1,M2)
%     T=zeros(M1*M2,M1*M2);
    u=reshape(u,[2*M2-1,2*M1-1]);
    u=1/2*(u(:,M1:end)+conj(u(end:-1:1,M1:-1:1)));
    u0=1/2*(u(M2:end,1)+conj(u(M2:-1:1,1)));
    for k=1:M1
     T(M2*(k-1)+1:k*M2,M2*(k-1)+1:k*M2)=toeplitz(u0);
    end
    for i=1:M1-1
     T0=toeplitz(u(M2:-1:1,i+1),u(M2:2*M2-1,i+1));
     for k=1:M1-i
         T(M2*(k-1)+1:k*M2,M2*(k+i-1)+1:(k+i)*M2)=T0;
         T(M2*(k+i-1)+1:(k+i)*M2,M2*(k-1)+1:k*M2)=T0';
     end
    end
end