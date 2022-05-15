clear variables
close all
clc

% Modulation Parameters
Q = 2; 
M = 2^Q; % 16QAM modulation
Es = 10;  % Average symbol energy

% Noise Parameters
EbN0_dB = 1:1:15; % SNR per bit (dB)
EsN0_dB = EbN0_dB + 10*log10(Q); % SNR per symbol (dB)
N0 = (Es/10).^(EsN0_dB/10);

N_train=20000;
N_test=1000000;

Y_train=randi(2^Q,1,N_train)';
Y_test=randi(2^Q,1,N_test)';

Xt_train=dec2bin(Y_train-1);
Xr_train=2*bin2dec(Xt_train(:,1:Q/2))-2^(Q/2)+1;
Xi_train=2*bin2dec(Xt_train(:,Q/2+1:end))-2^(Q/2)+1;
X_train=Xr_train+j*Xi_train;
X_train_Size = size(X_train);

Xt_test=dec2bin(Y_test-1);
Xr_test=2*bin2dec(Xt_test(:,1:Q/2))-2^(Q/2)+1;
Xi_test=2*bin2dec(Xt_test(:,Q/2+1:end))-2^(Q/2)+1;
X_test=Xr_test+j*Xi_test;
X_test_Size = size(Y_test);


TRAIN=[X_train,Y_train];
TEST=[X_test,Y_test];

save('TRAIN_fedrec_2.mat', 'TRAIN')
save('TEST_fedrec_2.mat', 'TEST')

Es = (X_train'*X_train)/N_train;
Eb = Es/Q;

figure
plot(X_train,'*')
title('X Train')

figure
plot(X_test,'* r')
title('X Test')

figure
semilogx(Y_train,'o')
title('Y Train')

figure
semilogx(Y_test,'o r')
title('y Test')


