clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output/';

% Load first dataset
load('data_bestactions1.mat')
bestaction1 = bestactions1;

best_weight1 = bestaction1(:, 1);
best_action1 = bestaction1(:, 2);
qValue1 = bestaction1(:, 4);

% Load second dataset (assuming it has the same structure and is named differently)
load('data_bestactions2.mat')
bestaction2 = bestactions2;

best_weight2 = bestaction2(:, 1);
best_action2 = bestaction2(:, 2);
qValue2 = bestaction2(:, 4);

figure(1)
% Plot first dataset
plot(best_weight1, qValue1, 'LineWidth', 2);
hold on;

% Plot second dataset
plot(best_weight2, qValue2, '--', 'LineWidth', 2);

title('Q Value vs Weight');
xlabel('Weight');
ylabel('Q Value');
legend('TD Learning', 'Q Learning');
grid on;

% Save the figure (optional)
saveas(gcf, fullfile(folder_path, 'comp_qvalue_weight_without_penalty_2.jpg'));

