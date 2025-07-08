clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output_v2/qvalue_update_analysis';


load('data_qvalue_update.mat')
data_qValue_update  = qvalueupdate;

update = data_qValue_update(:,1);
qValue = data_qValue_update(:,2);

figure;
plot(update, qValue, 'b-', 'LineWidth', 2); % Plot for first Q value update
xlabel('Update Number');
ylabel('Q Value');
legend('(39, 1)');
title('State Q Value Over Update Number');
grid on;

saveas(gcf, fullfile(folder_path, 'qvalue_update_analysis.jpg'));
