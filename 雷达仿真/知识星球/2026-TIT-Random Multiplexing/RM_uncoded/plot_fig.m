load("data_original.mat")

figure(1)
semilogy(SNR_dB, BER_ofdm, '-', 'LineWidth', 1);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
hold on
semilogy(SNR_dB, BER_otfs, '-', 'LineWidth', 1);
semilogy(SNR_dB, BER_afdm, '-', 'LineWidth', 1);
semilogy(SNR_dB, BER_rm, 'r-', 'LineWidth', 1);
semilogy(SNR_dB, BER_se, 'ro', 'LineWidth', 1);
ylim([8e-6, 0.2])
xlim([4, 24])
xticks(4:4:24)
legend('OFDM + CD-OAMP', 'OTFS + CD-OAMP', 'AFDM + CD-OAMP', 'RM + CD-MAMP', 'SE');
xlabel('SNR (dB)');
ylabel('BER');

figure(2)
semilogy(SNR_dB, BER_rm, 'r-', 'LineWidth', 1);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);
hold on;
semilogy(SNR_dB, BER_none, '-', 'LineWidth', 1);
semilogy(SNR_dB, BER_none_m, '-', 'LineWidth', 1);
semilogy(SNR_dB, BER_se, 'ro', 'LineWidth', 1);
ylim([8e-6, 0.2])
xlim([4, 24])
xticks(4:4:24)
legend('RM + CD-MAMP', 'No modulation + CD-OAMP', 'No modulation + CD-MAMP', 'SE');
xlabel('SNR (dB)');
ylabel('BER');

