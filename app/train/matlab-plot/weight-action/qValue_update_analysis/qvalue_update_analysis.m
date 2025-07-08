clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output_v2/qvalue_update_analysis';

load('data_qvalue_update1.mat')
data_qValue_update1  = qvalueupdate1;

load('data_qValue_update2.mat')
data_qValue_update2  = qvalueupdate2;

update1 = data_qValue_update1(:,1);
qValue1 = data_qValue_update1(:,2);

update2 = data_qValue_update2(:, 1);
qValue2 = data_qValue_update2(:, 2);

figure;
plot(update1, qValue1, 'b-', 'LineWidth', 2); % Plot for first Q value update
hold on;
plot(update2, qValue2, 'r-', 'LineWidth', 2); % Plot for second Q value update
hold off;
xlabel('Update Number');
ylabel('Q Value');
legend('(52, 1)', '(52, -1)');
title('State Q Value Over Update Number');
grid on;

% saveas(gcf, fullfile(folder_path, 'comp_qvalue_update_analysis__qinitial0_52.jpg'));
