%% Modulation %%
function [x_data,x] = Get_modulation(MT,Nx,x_data)

%     load x_data.mat;
%       x_data = randi([0 1],2*Nx,1);
    
% if MT == 1 %%% BPSK
%     x = (2*randi(2,N,M)-3);
      if MT == 2 %%% QPSK
           xx = x_data;
           btran = [];
          for i=1:2:2*Nx
              if xx(i) == 0 && xx(i+1) == 0
                  btra = (1+1i)/sqrt(2);
                  else if xx(i) == 0 && xx(i+1) == 1
                      btra = (1-1i)/sqrt(2);
                     else if xx(i) == 1 && xx(i+1) == 1
                           btra = (-1-1i)/sqrt(2);
                         else btra = (-1+1i)/sqrt(2);
                         end
                     end
             end
            btran = [btran;btra];
          end
      x = btran;
      end
% end

end















