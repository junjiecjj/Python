function MI = SF_Numerical_MI(PX,PS,X_input,Y_output,State)

    N_symbol              =  length(PX);
    Q_XY_given_S          =  zeros(N_symbol,N_symbol,N_symbol);                                 %初始化XYS的联合概率矩阵
    Q_Y_given_XS          =  zeros(N_symbol,N_symbol,N_symbol);                                 %初始化Y|XS的条件概率密度矩阵

    % 根据Gaussian信道模型Y=XS+N，N为N(0,1)噪声
    for  i1  =  1:N_symbol
        for i2  =  1:N_symbol
            for i3  =  1:N_symbol
                Q_Y_given_XS(i1,i2,i3)   =  1./sqrt(2*pi).*exp(-1/2.*(Y_output(i1)-State(i2).*X_input(i3)).^2);         
            end
        end
    end

    % 归一化概率，使其成为PDF 
    for is = 1:N_symbol        
        for ix = 1:N_symbol
            temp                     = sum(Q_Y_given_XS(:,is,ix));
            Q_Y_given_XS(:,is,ix)    = Q_Y_given_XS(:,is,ix) / temp;
        end
    end

    % 根据Q(X,Y|S)=Q(Y|XS)*Q(X), 生成联合概率密度函数
    for  iy  =  1:N_symbol
        for is  =  1:N_symbol
            for ix  =  1:N_symbol
                Q_XY_given_S(iy,is,ix)   =  Q_Y_given_XS(iy,is,ix)*PX(ix);         
            end
        end
    end

    Q_Y_Margin_S         =  sum(Q_XY_given_S,3);                        %对X维度积分，求Y的边缘PDF
    

 %% 利用公式I(X;Y)=H(Y)-H(Y|X)来求互信息
    H_y_given_XS         = zeros(N_symbol,1);

    for ix = 1:N_symbol
        for iy = 1:N_symbol
            for is = 1:N_symbol
                H_y_given_XS(is) = H_y_given_XS(is) - Q_XY_given_S(iy,is,ix) .* log2(Q_Y_given_XS(iy,is,ix));
            end
        end
    end

    H_y_given_XS(isnan(H_y_given_XS))  =  mean(H_y_given_XS(~isnan(H_y_given_XS)));
    H_Y_given_S                        =  -sum(Q_Y_Margin_S.*log2(Q_Y_Margin_S),1).';
    I_XY_given_S                       =  H_Y_given_S-H_y_given_XS;
    I_XY_given_S(isnan(I_XY_given_S))  =  0;
    MI                                 =  PS.'*I_XY_given_S;

end