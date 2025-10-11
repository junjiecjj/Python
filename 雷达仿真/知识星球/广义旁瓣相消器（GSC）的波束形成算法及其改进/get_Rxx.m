function Rxx=get_Rxx(s_rec,N,p,m)
    Rxx=zeros(m,m);
    x=1:m;
    for i=1:N
        for t=1:m
            x(t)=s_rec(t,i);
        end
        R=x'*x;
        Rxx=Rxx+R;
    end
    Rxx=Rxx/N;
end
