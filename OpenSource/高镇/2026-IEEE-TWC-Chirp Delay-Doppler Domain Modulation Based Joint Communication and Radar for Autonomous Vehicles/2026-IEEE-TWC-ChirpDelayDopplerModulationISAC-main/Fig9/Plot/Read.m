DDM_filename_IM = './DDM/mi_IM_';
TDM_filename_IM = './TDM/mi_IM_';
filename_end = '.csv';
QAM_order = [4 16 64];
SNR = -10:4:30;
DDM_data = zeros(3, 5, length(SNR));
TDM_data = zeros(3, 5, length(SNR));

for i = 1:length(QAM_order)
    filename = [DDM_filename_IM ...
                num2str(QAM_order(i)) ...
                filename_end];
    tmp = readmatrix(filename);
    DDM_data(i,:,:) = tmp(:,2:end);
end

for i = 1:length(QAM_order)
    filename = [TDM_filename_IM ...
                num2str(QAM_order(i)) ...
                filename_end];
    tmp = readmatrix(filename);
    TDM_data(i,:,:) = tmp(:,2:end);
end
save('IM',"TDM_data","DDM_data","SNR")