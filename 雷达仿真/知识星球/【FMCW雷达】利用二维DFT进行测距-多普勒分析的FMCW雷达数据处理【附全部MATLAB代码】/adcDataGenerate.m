% Function adcDataGenerate that simulates the data from the given structue
% radarparamets

function Rx=adcDataGenerate(radar_parameters,target_range,target_velocity)
        c = 3e8;%Speed of light in m/s
        N = radar_parameters.N;
        L = radar_parameters.L; 
        T = radar_parameters.T;
        B = radar_parameters.B;
        max_range = radar_parameters.max_range;
        f_start = radar_parameters.f_start;
        TRRI = radar_parameters.TRRI;
        S = radar_parameters.S;
        f_s = radar_parameters.f_s;
        T_s=radar_parameters.T_s; % Sampling Period
        R = target_range;
        v = target_velocity; 
        max_prop_time = 3 *max_range/c;%Maximum delay
        

        for Chi_idx=1:L %Chirp index for L Chirps
            
            for Sam_idx=1:N %Sample index for N Samples
               
                t = max_prop_time +(Sam_idx-1)*T_s;
               
                phase_Tx(Sam_idx,Chi_idx) = 2*pi*f_start*t+pi*S*t^2;

                %Phase  of the received Signal
                tau = 2*(R+v*t+v*TRRI*(Chi_idx-1))/c;
                
                phase_Rx(Sam_idx,Chi_idx) = 2 * pi * f_start *(t-tau) + pi * S * (t-tau)^2;
               
                %Phase of the mixed Signal/IF signal 
            
                phase_IF(Sam_idx,Chi_idx)=phase_Tx(Sam_idx,Chi_idx)- phase_Rx(Sam_idx,Chi_idx); 
        
                %The Mixed Signal/ IF signal
            
               Rx(Sam_idx,Chi_idx)= complex(cos(phase_IF(Sam_idx,Chi_idx)),sin(phase_IF(Sam_idx,Chi_idx))); %Complex command
            end
        end 
end
