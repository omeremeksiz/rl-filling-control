clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output_v2/action_state_analysis/';

load('data_qvalue.mat')
data_qvalue  = qvalueoutput;

weight=data_qvalue(:,1);
action=data_qvalue(:,2);

load('data_bestactions.mat')
bestaction = bestactions;

best_weight = bestaction(:, 1);
best_action = bestaction(:, 2);

%% Action vs Weight

figure(1)

idx_action_1 = (action == 1);
scatter(weight(idx_action_1), action(idx_action_1), 90, [0, 0, 1], 'o', 'filled', 'DisplayName', 'Coarse Feed (1)');
hold on;

idx_action_minus_1 = (action == -1);
scatter(weight(idx_action_minus_1), action(idx_action_minus_1), 90, [1, 0, 0], '^', 'filled', 'DisplayName', 'Fine Feed (-1)');

title('Action vs State');
xlabel('State');
ylabel('Action');
legend('show');
grid on;

% saveas(gcf, fullfile(folder_path, 'action_weight.jpg'));

%% Best Actions and Weight

figure(2)

idx_action_1 = (best_action == 1);
scatter(best_weight(idx_action_1), best_action(idx_action_1), 90, [0, 0, 1], 'o', 'filled', 'DisplayName', 'Kaba Besleme (1)');
hold on;

idx_action_minus_1 = (best_action == -1);
scatter(best_weight(idx_action_minus_1), best_action(idx_action_minus_1), 90, [1, 0, 0], '^', 'filled', 'DisplayName', 'İnce Besleme (-1)');

% title('Best Action vs State')
% xlabel('State')
% ylabel('Action')
xlabel('Durum')
ylabel('Eylem')
ylim([-1 1])
legend('show');
grid on

% saveas(gcf, fullfile(folder_path, 'bestaction_state_10050.jpg'));