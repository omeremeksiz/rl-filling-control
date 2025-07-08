excel=readtable('/Users/omeremeksiz/Desktop/MasaUstu/R&L Works/data_v2/quantizationCheck.xlsx');

non_quantized_1=excel.non_quantized_1;
quantized_1=excel.quantized_1*4000;

non_quantized_2=excel.non_quantized_2;
quantized_2=excel.quantized_2*4000;

figure(1)
subplot(3,1,1)
plot(non_quantized_1,'--','LineWidth',2)
grid on
hold on
title('All Data')
plot(quantized_1)
legend('Non Quantized Data','Quantized Data','Location','southeast')

subplot(3,1,2)
plot(non_quantized_1,'--','LineWidth',2)
xlim([100 200])
grid on
hold on
title('Scaled x axis between 175-200')
plot(quantized_1)
xlim([175 200])
legend('Non Quantized Data','Quantized Data','Location','southeast')

subplot(3,1,3)
plot(non_quantized_1,'--','LineWidth',2)
xlim([100 200])
grid on
hold on
title('Scaled x axis between 100-150')
plot(quantized_1)
xlim([100 150])
legend('Non Quantized Data','Quantized Data','Location','southeast')

figure(2)
subplot(3,1,1)
plot(non_quantized_2,'--','LineWidth',2)
grid on
hold on
title('All Data')
plot(quantized_2)
legend('Non Quantized Data','Quantized Data','Location','southeast')

subplot(3,1,2)
plot(non_quantized_2,'--','LineWidth',2)
xlim([100 200])
grid on
hold on
title('Scaled x axis between 175-200')
plot(quantized_2)
xlim([175 200])
legend('Non Quantized Data','Quantized Data','Location','southeast')

subplot(3,1,3)
plot(non_quantized_2,'--','LineWidth',2)
xlim([100 200])
grid on
hold on
title('Scaled x axis between 100-150')
plot(quantized_2)
xlim([100 150])
legend('Non Quantized Data','Quantized Data','Location','southeast')

