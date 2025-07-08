clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output_v2/alpha_param';

load('data_bestactions_alpha01.mat')
data_bestactions_alpha01  = bestactions1;

weight_alpha01 = data_bestactions_alpha01(:,1);
qvalue_alpha01 = data_bestactions_alpha01(:,4);

load('data_bestactions_alpha02.mat')
data_bestactions_alpha02  = bestactions2;

weight_alpha02 = data_bestactions_alpha02(:,1);
qvalue_alpha02 = data_bestactions_alpha02(:,4);

load('data_bestactions_alpha03.mat')
data_bestactions_alpha03  = bestactions3;

weight_alpha03 = data_bestactions_alpha03(:,1);
qvalue_alpha03 = data_bestactions_alpha03(:,4);

load('data_bestactions_alpha04.mat')
data_bestactions_alpha04  = bestactions4;

weight_alpha04 = data_bestactions_alpha04(:,1);
qvalue_alpha04 = data_bestactions_alpha04(:,4);

load('data_bestactions_alpha05.mat')
data_bestactions_alpha05  = bestactions5;

weight_alpha05 = data_bestactions_alpha05(:,1);
qvalue_alpha05 = data_bestactions_alpha05(:,4);

figure(1)
plot(weight_alpha01, qvalue_alpha01, 'LineWidth', 2);
title('Q Value vs Weight');
xlabel('Weight');
ylabel('Q Value');
grid on;
hold on;

plot(weight_alpha02, qvalue_alpha02, 'LineWidth', 2);
plot(weight_alpha03, qvalue_alpha03, 'LineWidth', 2);
plot(weight_alpha02, qvalue_alpha04, 'LineWidth', 2);
plot(weight_alpha03, qvalue_alpha05, 'LineWidth', 2);


legend('\alpha = 0.1','\alpha = 0.2', '\alpha = 0.3', '\alpha = 0.4', '\alpha = 0.5', 'Location','northwest');

legendFontSize = 12;
legend('FontSize', legendFontSize);

hold off;

saveas(gcf, fullfile(folder_path, 'alpha_param_analysis.jpg'));