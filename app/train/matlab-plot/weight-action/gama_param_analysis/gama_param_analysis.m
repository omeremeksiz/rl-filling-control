clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output_v2/gama_param';

load('data_bestactions_gama09.mat')
data_bestactions_gama09  = bestactions1;

weight_gama09 = data_bestactions_gama09(:,1);
qvalue_gama09 = data_bestactions_gama09(:,4);

load('data_bestactions_gama099.mat')
data_bestactions_gama099  = bestactions2;

weight_gama099 = data_bestactions_gama099(:,1);
qvalue_gama099 = data_bestactions_gama099(:,4);

load('data_bestactions_gama0999.mat')
data_bestactions_gama0999  = bestactions3;

weight_gama0999 = data_bestactions_gama0999(:,1);
qvalue_gama0999 = data_bestactions_gama0999(:,4);

figure(1)
plot(weight_gama09, qvalue_gama09, 'LineWidth', 2);
title('Q Value vs Weight');
xlabel('Weight');
ylabel('Q Value');
grid on;
hold on;

plot(weight_gama099, qvalue_gama099, 'LineWidth', 2);
plot(weight_gama0999, qvalue_gama0999, 'LineWidth', 2);

legend('\gamma = 0.9', '\gamma = 0.99', '\gamma = 0.999', 'Location','southeast');

legendFontSize = 12;
legend('FontSize', legendFontSize);

hold off;

saveas(gcf, fullfile(folder_path, 'gama_param_analysis.jpg'));