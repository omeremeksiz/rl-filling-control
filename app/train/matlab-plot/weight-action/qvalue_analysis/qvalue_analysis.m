clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output_v2/qvalue_analysis';

load('data_bestactions.mat')
bestaction = bestactions;

best_weight = bestaction(:, 1);
best_action = bestaction(:, 2);
qValue=bestaction(:,4);

figure(1)
plot(best_weight, qValue, 'LineWidth', 2);
title('Q Value vs Weight');
xlabel('Weight');
ylabel('Q Value');
grid on;

% saveas(gcf, fullfile(folder_path, 'qvalue_weight_with_penalty.jpg'));

