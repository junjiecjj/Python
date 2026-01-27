function Gain_RDM = DetermineGain(X, TargetList_ort, Para)
%  obtain channel gain of current frame
% 
%  input
%       X                   RDM × Nr.tot
%       TargetList_ort      used to extract velocity and range index
%  output
%       Gain_RDM            channel gain of current frame, Nt.tot × Nr.tot
%%
Gain_RDM = zeros(Para.Nt.tot, Para.Nr.tot);
if isempty(TargetList_ort(1).range)
    error('NumTarget = 0');
else
    NumTarget = length(TargetList_ort);
end
% % obtain all the label of the newest TargetList
% AllLabel_cell = {TargetList_ort(:).label};
% AllLabel_array = squeeze(char((AllLabel_cell )));

for i = 1:NumTarget
    if strcmp(TargetList_ort(i).label, 'A_0')
        range_average = round(TargetList_ort(i).range_average);
        velocity_average = round(TargetList_ort(i).velocity_average);
        break
    end
end

velocity_all = mod(velocity_average - 1 + (0:Para.N_c/Para.Nt.tot:Para.N_c-Para.N_c/Para.Nt.tot), ...
                   Para.N_c) + 1;

for i = 1:Para.Nt.tot
    for j = 1:Para.Nr.tot
        Gain_RDM(i,j) = X(range_average, ...
                        velocity_all(i), ...
                        j);
    end
end
end