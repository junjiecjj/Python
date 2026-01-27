function f = GD_ZR(Nit, rho, alpha_ini, sigma, count, count_max, A, x)
f = zeros(Nit, 1);
for i_it = 1:Nit
    alpha = alpha_ini;
    nabla = Nabla(A, x);
    while Quadratic(A, x-alpha*nabla) >  (Quadratic(A, x)+alpha*rho*nabla) 
        alpha = alpha/1.2;
    end
    x_2 = x - alpha*nabla;
    if Nabla(A, x_2)'*Nabla(A, x_2) < sigma*Nabla(A, x)'*Nabla(A, x)% 不仅要满足直线下方，还要满足梯度的绝对值减小
        x = x_2;
    else
        if Nabla(A, x_2)'*Nabla(A, x) < 0 % 点2的梯度与点1的梯度相反
            x_3 = (x + x_2)/2;
            while Nabla(A, x_3)'*Nabla(A, x_3) > sigma*Nabla(A, x)'*Nabla(A, x)      
                if count > count_max
                    break;
                end
                if Nabla(A, x_3)'*Nabla(A, x) < 0
                    x_3 = (x + x_3)/2;
                else
                    x_3 = (x_2 + x_3)/2;
                end
                count = count + 1;
            end
            x = x_3;
        else
            x_3 = 2*x_2 - x;
            while Quadratic(A, x_3) >  Quadratic(A, x_2) 
                x_3 = (x_2 + x_3)/2;
            end
            while Nabla(A, x_3)'*Nabla(A, x_3) > sigma*Nabla(A, x)'*Nabla(A, x)
                if count > count_max
                    break;
                end
                if Nabla(A, x_3)'*Nabla(A, x) < 0
                    x_3 = (x_2 + x_3)/2;
                else
                    x_3 = (x_3 - x_2)/2 +x_3;
                end
                count = count + 1;
            end
            x = x_3;
        end
    end
    count = 1;
    f(i_it) = Quadratic(A, x);
end    
end