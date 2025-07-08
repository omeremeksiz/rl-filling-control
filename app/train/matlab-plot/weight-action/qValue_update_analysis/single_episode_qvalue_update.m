clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output_v2/qvalue_update_analysis';


load('data_single_episode_qvalue_update.mat')
data_single_episode_qValue_update  = singleepisodeqvalueupdate;

weight = data_single_episode_qValue_update(:,1);
qValue = data_single_episode_qValue_update(:,4);

figure;
plot(weight, qValue, 'b-', 'LineWidth', 2); % Plot for first Q value update
xlabel('State');
ylabel('Q Value');
legend('Episode 2');
title('Single Episode States Q Value Update');
grid on;

% saveas(gcf, fullfile(folder_path, 'qvalue_update_analysis.jpg'));
