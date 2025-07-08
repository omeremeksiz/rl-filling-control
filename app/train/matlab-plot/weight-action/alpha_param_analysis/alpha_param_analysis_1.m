clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output/';

load('data_bestactions_alpha01.mat')
data_bestactions_alpha01  = bestactions1;

weight_alpha01 = data_bestactions_alpha01(:,1);
qvalue_alpha01 = data_bestactions_alpha01(:,4);

load('data_bestactions_alpha005.mat')
data_bestactions_alpha005  = bestactions2;

weight_alpha005 = data_bestactions_alpha005(:,1);
qvalue_alpha005 = data_bestactions_alpha005(:,4);

load('data_bestactions_alpha001.mat')
data_bestactions_alpha001  = bestactions3;

weight_alpha001 = data_bestactions_alpha001(:,1);
qvalue_alpha001 = data_bestactions_alpha001(:,4);

load('data_bestactions_alpha0001.mat')
data_bestactions_alpha0001  = bestactions4;

weight_alpha0001 = data_bestactions_alpha0001(:,1);
qvalue_alpha0001 = data_bestactions_alpha0001(:,4);

figure(1)
plot(weight_alpha0001, qvalue_alpha0001, 'LineWidth', 2);
title('Q Learning Q Value vs Weight');
xlabel('Weight');
ylabel('Q Value');
grid on;
hold on;

plot(weight_alpha001, qvalue_alpha001, 'LineWidth', 2);
plot(weight_alpha005, qvalue_alpha005, 'LineWidth', 2);
plot(weight_alpha01, qvalue_alpha01, 'LineWidth', 2);

legend('\alpha = 0.001','\alpha = 0.01', '\alpha = 0.05', '\alpha = 0.1', 'Location','northwest');

legendFontSize = 12;
legend('FontSize', legendFontSize);

hold off;

saveas(gcf, fullfile(folder_path, 'q_alpha_param_analysis.jpg'));