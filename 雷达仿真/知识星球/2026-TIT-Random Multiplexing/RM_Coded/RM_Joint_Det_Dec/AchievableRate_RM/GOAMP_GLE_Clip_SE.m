function rho=GOAMP_GLE_Clip_SE(delta, clip, var_n, v_nle)

precious = 10^-3;
v_mmse=zeros(length(var_n), length(v_nle));
S=20000;

v_x =1.0;
v_z=v_x;

for jj=1:length(var_n)
    v_n=var_n(jj);
    %source
    for  ii=1:length(v_nle)
        jj
        ii
        vz_dem = v_nle(ii);
        v1 = 10^-20;
        v2 = 1;
        iter=1;
        while(abs(v1-v2)>precious)
         
             z_re  = normrnd(0, sqrt(v_z-vz_dem), [round(S*delta), 1]);
            z_im = normrnd(0, sqrt(v_z-vz_dem), [round(S*delta), 1]);
            z      = (z_re + z_im * 1i) / sqrt(2);
            n_re = normrnd(0, sqrt(vz_dem), [round(S*delta), 1]);
            n_im = normrnd(0, sqrt(vz_dem), [round(S*delta), 1]);
            n = (n_re + n_im * 1i) / sqrt(2);
            z_n = z + n;
             %clipping
             z_n_re=real(z_n);
             z_n_im=imag(z_n);
             z_n_re(z_n_re>clip)   =  clip;
             z_n_re(z_n_re<-clip)  =-clip;
             z_n_im(z_n_im>clip) =  clip;
             z_n_im(z_n_im<-clip)=-clip;
             y=z_n_re+1i.*z_n_im;
              %NLD--psi
            vz_nle_post_real = GOAMP_Declip_SE_vec(clip, real(z),   vz_dem/2, real(y),   v_n/2);
            vz_nle_post_img = GOAMP_Declip_SE_vec(clip, imag(z), vz_dem/2, imag(y), v_n/2);
            
            vz_nle_post = vz_nle_post_real + vz_nle_post_img;
            vz_nle_post=delta * vz_nle_post + (1-delta)* vz_dem;
             v_mmse(jj, ii) = vz_nle_post;
            
            if mod(iter,2)==1
                v1=v_mmse(jj, ii);
                iter=iter+1;
            else
                v2=v_mmse(jj, ii);
                iter=iter+1;
            end
        end
        %v_mmse(jj, ii)
    end
end
    rho=v_mmse;
end

%% LE_SE
function [v_post, vz_post] = GOAMP_LE_SE(N, v, dia, v_z)
Dia        = 1 ./ (v_z ./ dia.^2 + v);
v_post   = v - v^2 * sum(Dia) / N;
Dia        = 1 ./ (dia.^2 /v_z  + 1/v);
vz_post = mean(dia.^2 .* Dia);
end