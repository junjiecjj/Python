%--------------------------------------------------------------------------
%       Simulación Radar - PRUEBA DE MULTIPLES BLANCOS VISUALMENTE
%--------------------------------------------------------------------------
% Autores: Miguel Carralero Lanchares y Franciso Orcha Kovacs

clear; close all; clc;

%% --- Parámetros Iniciales y Constantes ---
Pdg = 1e5;          % Potencia Pico (W)
Gant_dB = 30;       % Ganancia Antena (dB)
f = 9e9;            % Frecuencia (Hz)
tau = 1e-6;         % Ancho Pulso (s)
Ls_dB = 5;          % Pérdidas Sistema (dB)
Fn_dB = 4;          % Factor Ruido Rx (dB)
Pfa = 1e-6;         % Prob. Falsa Alarma
Pd = 0.9;           % Prob. Detección
RCS_mean_calc = 1;  % RCS media (m^2)
theta_bw_deg = 3.0; % Ancho de Haz de la antena (grados)

% Constantes físicas y derivadas
c = physconst('LightSpeed'); 
k = physconst('Boltzmann'); 
T0 = 290;
Gant = 10^(Gant_dB/10); 
lambda = c/f; 
Ls = 10^(Ls_dB/10);
Fn = 10^(Fn_dB/10); 
Bn = 1/tau; 
theta_bw_rad = deg2rad(theta_bw_deg);

%% • Inicialmente debe hacerse un cálculo del alcance máximo (Rmax) para la Pdg,
%    Gant, Lsistema, anchura pulso, SN según la Pfa y Pd y el valor medio de sección
%    radar de los blancos (RCS_mean_calc).
% •Incorporación blancos fluctuantes (Nota: Rmax se calcula para RCS_mean_calc no fluctuante,
%  pero la Pd real variará debido a la fluctuación de RCS implementada más adelante.
%  Para un cálculo de Rmax riguroso con blancos fluctuantes, se necesitaría
%  ajustar SNR_minima_requerida según el modelo Swerling).
if exist('shnidman', 'file') && ~isempty(which('shnidman'))
    SNR_req_dB = shnidman(Pd, Pfa); 
else
    warning('shnidman no encontrada. Usando aproximación Albersheim N=1 (no fluctuante).');
    A_alb = log(0.62 / Pfa); 
    B_alb = log(Pd / (1 - Pd));
    SNR_req_dB = A_alb + 0.12 * A_alb * B_alb + 1.7 * B_alb;
end
SNR_minima_requerida = 10^(SNR_req_dB / 10);

Ntot_valor_medio = k * T0 * Fn * Bn;

Rmax_num = Pdg * Gant^2 * lambda^2 * RCS_mean_calc;
Rmax_den = (4*pi)^3 * Ls * Ntot_valor_medio * SNR_minima_requerida;
Rmax = (Rmax_num / Rmax_den)^(1/4);
fprintf('Alcance Máximo (Rmax, basado en RCS_mean_calc no fluctuante): %.2f km\n', Rmax/1000);

% o Tendrán que indicar la resolución mínima y distancia mínima en función de la anchura de pulso
R_res = c * tau / 2; R_min = R_res;
fprintf('Resolución Mínima: %.2f m\nDistancia Mínima: %.2f m\n', R_res, R_min);

%% • A partir de ello:
% o Ajustarse la PRF se corresonda con Rmax,na...
Rmax_na = (4/3) * Rmax;
if Rmax >= Rmax_na;
    warning('Rmax >= Rmax_na.');
    Rmax_na = Rmax * 1.1;
end
PRI = 2 * Rmax_na / c;
PRF = 1 / PRI;
fprintf('Rmax_na: %.2f km, PRF: %.2f Hz, PRI: %.4f ms\n', Rmax_na/1000, PRF, PRI*1000);

% o Conocida la SNmínima ... Smin_receptor ... Prx_min ... Vrx_min ...
Smin_receptor = SNR_minima_requerida * Ntot_valor_medio;
Prx_min_antena = Smin_receptor * Ls;
fprintf('Smin_receptor: %.2e W, Prx_min_antena: %.2e W\n', Smin_receptor, Prx_min_antena);

% o Debe sacarse el Vumbral a partir de la Pfa
Vumbral_absoluto_envolvente = sqrt(Ntot_valor_medio * 2 * log(1/Pfa)); 
fprintf('Umbral de Voltaje de Envolvente Absoluto (Vumbral): %.2e V\n', Vumbral_absoluto_envolvente);


%% --- Preparación de la Simulación Dinámica ---
% • --- Ahora para N_TARGETS blancos móviles ---
% • Ahora sin cambiar Pdg, Gant, Lsistema, anchura pulso, Pfa y Pd habrá que sacar
%   N_TARGETS blancos en posición aleatoria, que puede estar fuera de ese Rmax (por
%   supuesto), y cuya sección radar también será aleatoria alrededor del valor medio
%   que se ha usado en el cálculo del alcance máximo.

sim_time = 180; 
scan_rate_deg = 72; 
scan_rate_rad = deg2rad(scan_rate_deg);
dt = PRI;
rng('shuffle');

% --- Definición para múltiples blancos ---
N_TARGETS = 5;
target_pos = zeros(N_TARGETS, 2); 
velocity = zeros(N_TARGETS, 2);
target_colors = lines(N_TARGETS); 

max_initial_range = Rmax_na * 1.3; 
min_initial_range_factor = 0.3; 

% Inicializar cada blanco
for k_target = 1:N_TARGETS
    start_angle_rad_k = rand * 2*pi;
    start_range_k = (R_min + Rmax_na * min_initial_range_factor) + rand * (max_initial_range - (R_min + Rmax_na * min_initial_range_factor));
    
    % Aumentar velocidades base y rango de aleatoriedad ---
    if start_range_k > Rmax_na
        speed_k = 600 + rand*300; % Rango: 600 a 900 m/s
    else
        speed_k = 400 + rand*250; % Rango: 400 a 650 m/s
    end
    target_pos(k_target, :) = [start_range_k*cos(start_angle_rad_k), start_range_k*sin(start_angle_rad_k)];
    
    % Dirección inicial siempre hacia el radar (origen) ---
    if norm(target_pos(k_target, :)) > 1e-3 % Evitar división por cero si empieza exactamente en el origen
        velocity_direction_k = -target_pos(k_target, :) / norm(target_pos(k_target, :)); 
    else % Si por casualidad empieza en el origen, darle una dirección aleatoria
        random_direction_angle_k = rand * 2*pi;
        velocity_direction_k = [cos(random_direction_angle_k), sin(random_direction_angle_k)];
    end
    velocity(k_target, :) = speed_k * velocity_direction_k;
    fprintf('Inicio Blanco %d ALEATORIO: R=%.1f km, Ang=%.1f deg, Vel=%.1f m/s, Dir Hacia Radar\n', k_target, start_range_k/1000, rad2deg(start_angle_rad_k), speed_k);
end

time_vec = 0:dt:sim_time; 
n_steps_alloc = length(time_vec); 
target_pos_history = cell(N_TARGETS, 1); 
for k_target = 1:N_TARGETS
    target_pos_history{k_target} = NaN(n_steps_alloc, 2); 
end
detections = []; 

% • Recomendado poner al menos dos pantallas... 
main_fig = figure('Position',[50 50 1700 750]); 
display_range_limit_initial = Rmax_na*1.2; 
circ_coords_theta=linspace(0,2*pi,100);

% --- Definición de posiciones y tamaños de subplots [izquierda, abajo, ancho, alto] ---
pos_real_plot   = [0.05, 0.40, 0.43, 0.55]; 
pos_ppi_plot    = [0.53, 0.40, 0.43, 0.55]; 
pos_ascope_plot = [0.25, 0.05, 0.50, 0.28]; 

% --- Pantalla de Posición Real (subplot en pos_real_plot) ---
h_ax_real=axes('Parent', main_fig, 'Position', pos_real_plot); 
title(h_ax_real,'Posición Real'); 
xlabel(h_ax_real,'X (m)'); 
ylabel(h_ax_real,'Y (m)');
axis(h_ax_real,'equal'); 
grid(h_ax_real, 'on'); 
hold(h_ax_real, 'on'); 
plot(h_ax_real,0,0,'k^','MarkerSize',8,'DisplayName','Radar');
plot(h_ax_real,Rmax*cos(circ_coords_theta),Rmax*sin(circ_coords_theta),'--b','LineWidth',0.5,'DisplayName','Rmax (ref)');
plot(h_ax_real,Rmax_na*cos(circ_coords_theta),Rmax_na*sin(circ_coords_theta),'--r','LineWidth',0.5,'DisplayName','Rmax_{na}');
fill(h_ax_real,[Rmax*cos(circ_coords_theta),fliplr(Rmax_na*cos(circ_coords_theta))],[Rmax*sin(circ_coords_theta),fliplr(Rmax_na*sin(circ_coords_theta))],'y','FaceAlpha',0.1,'EdgeColor','none','DisplayName','Zona Ciega');
% --- Handles para múltiples trayectorias y posiciones ---
h_target_real_lines = gobjects(N_TARGETS, 1);
h_target_curr_pos = gobjects(N_TARGETS, 1);
for k_target = 1:N_TARGETS
    h_target_real_lines(k_target)=plot(h_ax_real,NaN,NaN,'-','Color', target_colors(k_target,:),'LineWidth',1.5,'DisplayName',sprintf('Blanco %d', k_target));
    h_target_curr_pos(k_target)=plot(h_ax_real,NaN,NaN,'o','Color', target_colors(k_target,:),'MarkerSize',6,'MarkerFaceColor',target_colors(k_target,:));
end
legend(h_ax_real,'Location','northwest');
axis(h_ax_real,display_range_limit_initial*[-1 1 -1 1]);

% --- Pantalla PPI (subplot en pos_ppi_plot) ---
ax_ppi=polaraxes('Parent',main_fig, 'Position', pos_ppi_plot); 
title(ax_ppi,'Radar PPI'); 
ax_ppi.ThetaZeroLocation='top'; 
ax_ppi.ThetaDir='clockwise';
ax_ppi.RLim=[0 display_range_limit_initial];
ax_ppi.Color=[0 0.1 0.15]; 
ax_ppi.GridColor=[0.5 1 0.5]; 
ax_ppi.RColor=[0.5 1 0.5]; 
ax_ppi.ThetaColor=[0.5 1 0.5]; 
ax_ppi.RAxis.Label.Color = 'k';     
ax_ppi.ThetaAxis.Label.Color = 'k'; 
ax_ppi.Title.Color = 'k';          
hold(ax_ppi, 'on');
polarplot(ax_ppi,circ_coords_theta,Rmax*ones(size(circ_coords_theta)),'--b','LineWidth',0.5,'HandleVisibility','off');
polarplot(ax_ppi,circ_coords_theta,Rmax_na*ones(size(circ_coords_theta)),'--r','LineWidth',0.5,'HandleVisibility','off');
h_scan_line_ppi=polarplot(ax_ppi,[NaN NaN],[0 display_range_limit_initial],'Color',[0.8 1 0.8],'LineWidth',1.5,'DisplayName','Scan');
h_detections_ppi_plot=polarplot(ax_ppi,NaN,NaN,'o','Color',[1 0.8 0.8],'MarkerSize',5,'MarkerFaceColor',[1 0.5 0.5],'DisplayName','Detecciones');
ppi_r_axis=ax_ppi.RAxis;
ppi_r_axis.Label.String='Rango (km)';
ppi_r_axis.Label.Color = 'w';
ppi_r_axis.TickLabelFormat='%g';
ppi_r_axis.TickValues=linspace(0,display_range_limit_initial,5); 
ppi_r_axis.TickLabels=string(round(ppi_r_axis.TickValues/1000));
legend(ax_ppi,'TextColor','w','Location','southoutside','Orientation','horizontal');


% --- Pantalla Tipo 1: A-SCOPE ---
h_ax_ascope = axes('Parent', main_fig, 'Position', pos_ascope_plot); 
title(h_ax_ascope,'Eco A-Scope (Amplitud vs Distancia)');
xlabel(h_ax_ascope,'Distancia (km)'); 
ylabel(h_ax_ascope,'Amplitud Envolvente (V)');
hold(h_ax_ascope, 'on');
grid(h_ax_ascope, 'on');
set(h_ax_ascope, 'Color', 'k'); 
set(h_ax_ascope, 'XColor', 'k', 'YColor', 'k', 'GridColor', [0.4 0.4 0.4]); 
h_ax_ascope.XLabel.Color = 'k'; 
h_ax_ascope.YLabel.Color = 'k'; 
h_ax_ascope.Title.Color = 'k';  

h_ascope_plot = plot(h_ax_ascope, NaN, NaN, 'g-', 'LineWidth', 1.5); 
h_ascope_threshold = yline(h_ax_ascope, Vumbral_absoluto_envolvente, '--r', 'LineWidth', 1);
set(h_ascope_threshold, 'DisplayName', 'V_{umbral}');
ascope_range_display_km = Rmax_na / 1000;
xlim(h_ax_ascope, [0 ascope_range_display_km]);
num_ascope_points = 500;
ascope_range_bins_km = linspace(0, ascope_range_display_km, num_ascope_points);
ascope_base_noise_amplitude = sqrt(Ntot_valor_medio/2); 
ascope_ylim_max = Vumbral_absoluto_envolvente * 5; 
ylim(h_ax_ascope, [0 ascope_ylim_max]);


%% --- Bucle Principal de Simulación ---
% • Ahora para N_TARGETS blancos móviles
fprintf('\nIniciando simulación continua...\n');
plot_update_interval = 100; 
plot_pause_duration = 0.001;
pulse_counter = 0; 

% ---Bucle while para funcionamiento continuo hasta cerrar figura ---
while ishandle(main_fig)
    pulse_counter = pulse_counter + 1;
    current_time = (pulse_counter - 1) * dt; 
    
    antenna_pointing_angle = mod(scan_rate_rad * current_time, 2*pi);
    
    % --- Variables para almacenar datos del A-Scope para el pulso actual ---
    ascope_picos_amplitud = [];
    ascope_picos_r_aparente_km = [];

    % --- Bucle sobre cada blanco ---
    for k_target = 1:N_TARGETS
        % Mover Blanco k
        target_pos(k_target, :) = target_pos(k_target, :) + velocity(k_target, :) * dt; 
        
        if pulse_counter <= n_steps_alloc 
             target_pos_history{k_target}(pulse_counter, :) = target_pos(k_target, :);
        else 
            target_pos_history{k_target} = [target_pos_history{k_target}(2:end,:); target_pos(k_target,:)];
        end

        R_target_real_k = norm(target_pos(k_target, :)); 
        angle_target_real_k = atan2(target_pos(k_target, 2), target_pos(k_target, 1));
        
        % --- Reiniciar el blanco si se aleja demasiado ---
        if R_target_real_k > max_initial_range * 1.5 
            fprintf('Blanco %d reiniciado por alejarse (R=%.1f km)...\n', k_target, R_target_real_k/1000);
            start_angle_rad_k = rand * 2*pi;
            start_range_k = (Rmax_na * 0.8) + rand * (max_initial_range - (Rmax_na * 0.8)); 
            % Aumentar velocidades base y rango de aleatoriedad al reiniciar ---
            if start_range_k > Rmax_na; speed_k = 600 + rand*300; else; speed_k = 400 + rand*250; end
            target_pos(k_target, :) = [start_range_k*cos(start_angle_rad_k), start_range_k*sin(start_angle_rad_k)];
            % Dirección del blanco reiniciado siempre hacia el radar ---
            if norm(target_pos(k_target, :)) > 1e-3
                velocity_direction_k = -target_pos(k_target, :) / norm(target_pos(k_target, :));
            else
                random_direction_angle_k = rand * 2*pi;
                velocity_direction_k = [cos(random_direction_angle_k), sin(random_direction_angle_k)];
            end
            velocity(k_target, :) = speed_k * velocity_direction_k;
            
            target_pos_history{k_target}(:,:) = NaN; 
            if pulse_counter <= n_steps_alloc
                target_pos_history{k_target}(pulse_counter, :) = target_pos(k_target, :);
            else 
                 target_pos_history{k_target}(end, :) = target_pos(k_target, :); 
            end
            fprintf('Nuevo inicio Blanco %d: R=%.1f km, Ang=%.1f deg, Vel=%.1f m/s, Dir Hacia Radar\n', k_target, start_range_k/1000, rad2deg(start_angle_rad_k), speed_k);
            continue; 
        end

        % •Incorporación blancos fluctuantes (Swerling I/II)
        RCS_instantanea_k = exprnd(RCS_mean_calc); 

        % • Debe ahora calcularse la potencia recibida, tensión recibida
        Prx_actual_antena_k = 0; S_actual_receptor_k = 0; is_target_illuminated_k = false;
        V_s_envolvente_pura_k = 0; 

        angular_difference_beam_k = abs(wrapToPi(antenna_pointing_angle - angle_target_real_k));
        if angular_difference_beam_k < (theta_bw_rad / 2) && R_target_real_k > R_min
            is_target_illuminated_k = true;
            Prx_num_calc_k = Pdg * Gant^2 * lambda^2 * RCS_instantanea_k;
            Prx_den_calc_k = (4*pi)^3 * R_target_real_k^4;
            Prx_actual_antena_k = Prx_num_calc_k / Prx_den_calc_k;
            S_actual_receptor_k = Prx_actual_antena_k / Ls;
            V_s_envolvente_pura_k = sqrt(S_actual_receptor_k); 
        end
        
        % • Debe calcularse el tiempo exacto que tarda la señal en llegar...
        t_vuelo_ida_vuelta_k = 2 * R_target_real_k / c;

        % • Debe añadirse ruido aleatorio... y sumarse a la señal
        sigma_ruido_componente_k = sqrt(Ntot_valor_medio / 2);
        v_n_I_k = randn() * sigma_ruido_componente_k; v_n_Q_k = randn() * sigma_ruido_componente_k;
        fase_senal_aleatoria_k = 2*pi*rand;
        v_total_I_k = V_s_envolvente_pura_k * cos(fase_senal_aleatoria_k) + v_n_I_k;
        v_total_Q_k = V_s_envolvente_pura_k * sin(fase_senal_aleatoria_k) + v_n_Q_k;
        V_envolvente_total_instantanea_k = sqrt(v_total_I_k^2 + v_total_Q_k^2);

        % • Hacer la comparación celda a celda... con respecto al Vumbral...
        % • Los blancos que estén en zona ciega... no salen en el radar.
        is_in_physical_blind_zone_k = (R_target_real_k >= Rmax && R_target_real_k < Rmax_na);
        
        can_attempt_detection_k = is_target_illuminated_k && ...
                                ~is_in_physical_blind_zone_k && ...
                                R_target_real_k > R_min;
        
        if can_attempt_detection_k
            ascope_picos_amplitud(end+1) = V_envolvente_total_instantanea_k;
            ascope_picos_r_aparente_km(end+1) = mod(R_target_real_k, Rmax_na)/1000;

            if V_envolvente_total_instantanea_k > Vumbral_absoluto_envolvente
                if t_vuelo_ida_vuelta_k < PRI
                    R_aparente_ppi_k = mod(R_target_real_k, Rmax_na);
                    detections(end+1, :) = [current_time, R_aparente_ppi_k, antenna_pointing_angle];
                end
            end
        end
    end % Fin del bucle de blancos

    % --- Actualización de Gráficas ---
    if mod(pulse_counter, plot_update_interval)==0
        % --- Actualizar Pantalla Posición Real ---
        for k_target = 1:N_TARGETS
            valid_hist_idx = ~isnan(target_pos_history{k_target}(:,1));
            if any(valid_hist_idx)
                set(h_target_real_lines(k_target),'XData',target_pos_history{k_target}(valid_hist_idx,1),'YData',target_pos_history{k_target}(valid_hist_idx,2));
            else
                set(h_target_real_lines(k_target),'XData',NaN,'YData',NaN); 
            end
            set(h_target_curr_pos(k_target),'XData',target_pos(k_target,1),'YData',target_pos(k_target,2));
        end
        
        % --- Actualizar PPI ---
        set(h_scan_line_ppi,'ThetaData',[antenna_pointing_angle antenna_pointing_angle]);
        if ~isempty(detections)
            scan_duration=360/scan_rate_deg; display_time_window=scan_duration*1.5; 
            valid_detections_to_plot=detections(detections(:,1)>=(current_time-display_time_window),:);
            if ~isempty(valid_detections_to_plot)
                set(h_detections_ppi_plot,'ThetaData',valid_detections_to_plot(:,3),'RData',valid_detections_to_plot(:,2));
                max_r_detected=max(valid_detections_to_plot(:,2));
                current_ppi_r_limit = ax_ppi.RLim(2);
                needed_ppi_r_limit=max([display_range_limit_initial,max_r_detected*1.05, Rmax_na*1.05]);
                 if current_ppi_r_limit < needed_ppi_r_limit || (max_r_detected < current_ppi_r_limit * 0.8 && current_ppi_r_limit > display_range_limit_initial)
                    ax_ppi.RLim(2)=needed_ppi_r_limit;
                    ppi_r_axis.TickValues=linspace(0,needed_ppi_r_limit,5);
                    ppi_r_axis.TickLabels=string(round(ppi_r_axis.TickValues/1000));
                 end
            else; set(h_detections_ppi_plot,'ThetaData',NaN,'RData',NaN); end
        else; set(h_detections_ppi_plot,'ThetaData',NaN,'RData',NaN); end
        
        current_max_real_dim_all_targets = 0;
        for k_target=1:N_TARGETS
             valid_hist_idx = ~isnan(target_pos_history{k_target}(:,1));
             if any(valid_hist_idx)
                max_coord_current_target_hist = max(abs(target_pos_history{k_target}(valid_hist_idx,:)), [], 'all');
                current_max_real_dim_all_targets = max(current_max_real_dim_all_targets, max_coord_current_target_hist);
             end
        end
        if current_max_real_dim_all_targets > 0
            current_xlim_real_val = get(h_ax_real, 'XLim');
            if current_max_real_dim_all_targets > current_xlim_real_val(2) || current_max_real_dim_all_targets < display_range_limit_initial 
                axis(h_ax_real, max(current_max_real_dim_all_targets * 1.1, display_range_limit_initial) * [-1 1 -1 1]);
            end
        else 
             axis(h_ax_real,display_range_limit_initial*[-1 1 -1 1]); 
        end

        % --- Actualizar pantalla Tipo 1: A-SCOPE (con múltiples picos si hay) ---
        ascope_y_data = ones(1, num_ascope_points) * ascope_base_noise_amplitude .* (1 + 0.2*rand(1, num_ascope_points)); 
        if ~isempty(ascope_picos_r_aparente_km)
            for p_idx = 1:length(ascope_picos_r_aparente_km)
                r_pico_km = ascope_picos_r_aparente_km(p_idx);
                amp_pico = ascope_picos_amplitud(p_idx);
                if r_pico_km <= ascope_range_display_km 
                    [~, idx_blanco_ascope] = min(abs(ascope_range_bins_km - r_pico_km));
                    ancho_pulso_display_ascope = max(1, round( (R_res/1000) / (ascope_range_display_km/num_ascope_points) )); 
                    idx_start = max(1, idx_blanco_ascope - floor(ancho_pulso_display_ascope/2));
                    idx_end = min(num_ascope_points, idx_blanco_ascope + ceil(ancho_pulso_display_ascope/2)-1);
                    
                    if idx_start <= idx_end 
                        num_pts_subida = ceil((idx_end-idx_start+1)/2);
                        if num_pts_subida > 0
                            pulse_shape_subida = linspace(ascope_y_data(idx_start), amp_pico, num_pts_subida);
                            ascope_y_data(idx_start : idx_start+num_pts_subida-1) = max(ascope_y_data(idx_start : idx_start+num_pts_subida-1), pulse_shape_subida);
                            
                            num_pts_bajada = (idx_end-idx_start+1) - num_pts_subida;
                            if num_pts_bajada > 0
                                pulse_shape_bajada = linspace(amp_pico, ascope_y_data(idx_end), num_pts_bajada+1);
                                ascope_y_data(idx_start+num_pts_subida : idx_end) = max(ascope_y_data(idx_start+num_pts_subida : idx_end), pulse_shape_bajada(2:end));
                            end
                        end
                    end
                end
            end
        end
        set(h_ascope_plot, 'XData', ascope_range_bins_km, 'YData', ascope_y_data);
        
        drawnow limitrate; 
        pause(plot_pause_duration);
    end

    % --- Condición de Parada de la Simulación (solo por tiempo o cerrar ventana) ---
    if current_time >= sim_time && sim_time > 0 
        fprintf('Tiempo de simulación (%.1f s) alcanzado. Terminando.\n', sim_time);
        break; 
    end
end % Fin del Bucle Principal (while)

if ishandle(main_fig)
    fprintf('Simulación detenida por el usuario o fin de tiempo.\n');
else
    fprintf('Ventana de simulación cerrada por el usuario.\n');
end