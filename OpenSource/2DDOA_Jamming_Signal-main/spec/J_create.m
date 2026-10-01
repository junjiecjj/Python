function J=J_create(Mx,My)
J=zeros(Mx^2*My^2,(2*Mx-1)*(2*My-1));
    for i=1:2*Mx-1
        kx=i-Mx;
        for j=1:2*My-1
            ky=j-My;
            J(:,(i-1)*(2*My-1)+j)=vec(kron(diag(ones(1,Mx-abs(kx)),kx),diag(ones(1,My-abs(ky)),ky)));
        end
    end
end