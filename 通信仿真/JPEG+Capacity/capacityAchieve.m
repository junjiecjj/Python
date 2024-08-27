

clc;
clear;



log_file = fopen('/home/jack/IPT-Pretrain-DataResults/log_new.txt','w+');
% log_fileq = fopen('/home/jack/IPT-Pretrain-DataResults/log_new_q.txt','w+');
% rootdir = '/home/jack/IPT-Pretrain-DataResults/Data/benchmark/';
% tempdir = '/home/jack/IPT-Pretrain-DataResults/CapacityAchievePicture/';

rootdir = '/home/jack/SemanticNoise_AdversarialAttack/Data/';
tempdir = '/home/jack/SemanticNoise_AdversarialAttack/Data/tmpdata/';

DataSetFolder = [
 "Mnist_test_png/"
% "B100/HR",
% "Set5/HR",
% "Set14/HR",
% "Urban100/HR",
];


tic;


compressRate = 1 : 0.1 : 1.2 ;
SNR = 10 : 1 : 20 ;


for d = 1:size(DataSetFolder, 1)
    fprintf("Dataset: %s\n", DataSetFolder(d));
    fprintf(log_file, '\n\nDataset: %-20s\n', DataSetFolder(d) );

    fprintf(log_file, '%-20s', '  Compress Rate = ');
    fprintf(log_file, '%.2f ', compressRate );
    fprintf(log_file, '\n');

    fprintf(log_file, '%-20s','  SNR = ');
    fprintf(log_file, '%.2f ', SNR );
    fprintf(log_file, '(dB) \n\n');

%     PSNR_resu = zeros(length(compressRate), length(SNR));
    files=dir( [rootdir, char(DataSetFolder(d)), '/*.png'] );
    for comp = compressRate
        fprintf("  comp = %.2d\n", comp);
        for snr = SNR
            fprintf("    snr = %.2d ", snr);
            PSNR = 0;
            folder = [tempdir, char(DataSetFolder(d)), '/comp=', num2str(comp), '/snr=', num2str(snr), '/'];
            mkdir(folder)
            cap = 0.5 * log2(1 + 10^(snr / 10));
            for i = 1:size(files, 1)
                file_name = files(i).name;
                im = imread([rootdir, char(DataSetFolder(d)), '/', file_name]);
                quality = get_correct_quanlity(im, numel(im)*comp*cap/8, i);
                % fprintf(log_fileq, 'comp = %6.2d, snr = %6.2d, file_name = %15s, quality = %d \n', comp, snr, file_name, quality);
                imwrite(im, [folder, file_name(1:end-3), 'jpg'], 'quality', quality);
                jm = imread([folder, file_name(1:end-3), 'jpg']);
                psnr = 10*log10(255^2/immse(im, jm));
                PSNR = PSNR + psnr;
                % fprintf(log_file, '%s, %.3f_%.1f, %s, %.2f\n', DataSetFolder{d}, comp, snr, file_name, psnr);
                % disp([DataSetFolder{d}, ' ', num2str(comp), ' ', num2str(snr), ' ', num2str(i/size(files, 1))])
            end
            PSNR = PSNR / size(files, 1);
            % PSNR_resu ;
            fprintf(log_file, '%.3f  ', PSNR);
            fprintf(', avg psnr %.3f  \n', PSNR);
        end
        fprintf(log_file, '\n');
    end
end
fclose(log_file);
% fclose(log_fileq);


function quality = get_correct_quanlity(im, file_size, idx)
l = 0;
r = 100;
temp_name = ['/tmp/', num2str(idx) ,'.jpg'];
while l < r
    m = ceil((l + r) / 2);
    imwrite(im, temp_name,'quality', m)
    dir_temp_name = dir(temp_name);
    if dir_temp_name.bytes <= file_size
        l = m;
    else
        r = m - 1;
    end
end
quality = l;
end






















