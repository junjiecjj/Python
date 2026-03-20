function [v, rho] = phi_star(sigma2, beta)
	rho_star = (1 - beta - sigma2+sqrt((1-beta - sigma2)^2 + 4*sigma2)) / (2 .* sigma2);
    v_star   = (1 / rho_star - sigma2) / beta;
	rho_max  = 1/sigma2;
    
    step1= (rho_star/20);
	rho1 = step1:step1:rho_star;
    
    step2= ((rho_max-rho_star)/20);
	rho2 = (rho_star+step2):step2:(rho_max-step2);
    
    step3= step2/100;
    rho3 = (rho_max-step2+step3):step3:(rho_max-step3);
    
    step4= step3/10;
    rho4 = (rho_max-step3+step4):step4:rho_max;
    
	v    = [1 ./ (1 + rho1), (1 ./ [rho2, rho3, rho4] - sigma2) ./ beta].';
    rho  = [rho1, rho2, rho3, rho4].';
    
    if(v_star>0.05)
        v = [v(:).', 0.05, 0.049, 0.0475, 0.045, 0.0425, 0.04];
        v = sort(v, 'descend');
        rho(v>v_star) = 1.0 ./ v(v>v_star) - 1.0;
        rho(v<=v_star) = 1.0 ./ (beta .* v(v<=v_star) + sigma2);
    end
    
    rho  = rho(v>=0.0);
    v    = v(v>=0.0);
    v    = v(:);
    rho  = rho(:);
    
    small_v = [1e-10:2e-10:9e-10, 1e-9:2e-9:9e-9, 1e-8:2e-8:9e-8, 1e-7:2e-7:9e-7, 1e-6:2e-6:9e-6, 1e-5:2e-5:9e-5, 1e-4:2e-4:9e-4, 1e-3:1e-3:9e-3, 1e-2:1e-2:9e-2, 1e-1:5e-2:(1-5e-2) 0.975 0.98 0.99];
    small_v = sort(small_v(small_v<min(v)), 'descend');
    v       = [v; small_v(:)];
    rho     = [rho; 1.0 ./ (beta .* small_v(:) + sigma2)];
    %rho = zeros(size(v));    
    %rho(v>v_star) = 1.0 ./ v(v>v_star) - 1.0;
    %rho(v<=v_star) = 1.0 ./ (beta .* v(v<=v_star) + sigma2);
end