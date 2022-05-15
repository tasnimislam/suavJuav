SNR = [1 5 10 15]

IID_DC_Blind_BER_mc_20_batch_20 = [0.086675 0.05812 0.03935 0.03207]
IID_BER_mc_20_batch_20 = [0.083306249 0.056249 0.036481 0.029962]

Non_IID_BER_mc_20_batch_20 = [0.09267 0.06467 0.04361 0.03752]
Non_IID_DC_Blind_BER_mc_20_batch_20 = [0.09538 0.067912 0.047512 0.039318]

Non_IID_BER_dc_comp_mc_5_batch_100000 = [0.15180299 0.14521 0.134905 0.1328099]
Non_IID_BER_original_mc_5_batch_100000 = [0.16072655 0.1513567 0.1490167 0.1394579]
figure()
plot(SNR, Non_IID_BER_original_mc_5_batch_100000)
hold on
plot(SNR, Non_IID_BER_dc_comp_mc_5_batch_100000)
xlabel('EbN0(dB)')
ylabel('BER')
legend('original', 'dc blind compensation')
title('Bit Error Rate vs EbN0(dB) for Non-IID User for 20000, 100000 dataset for mc = 5 for batch size = 10000')