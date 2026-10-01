function [d_c,c,c_p,c_n]=cent_ula(dx)
    D=ceil(length(dx)/2);
    dxp=dx(D:end);
    i=1;
    while 1
        if i>D
            break;
        end
        if dxp(i)~=i-1
            break;
        end
        i=i+1;
    end
    U=i-1;
    c=D-U+1:D+U-1;
    c_p=D:D+U-1;
    c_n=D:-1:D-U+1;
    d_c=1-U:U-1;
end