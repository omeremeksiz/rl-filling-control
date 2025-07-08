clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output';

load('data_qvalue_update1.mat')
data_qvalue_update1  = qvalueupdate1;

update1 = data_qvalue_update1(:,1);
qvalue1 = data_qvalue_update1(:,2);
gvalue1 = data_qvalue_update1(:,3);

load('data_qvalue_update2.mat')
data_qvalue_update2  = qvalueupdate2;

update2 = data_qvalue_update2(:,1);
qvalue2 = data_qvalue_update2(:,2);
gvalue2 = data_qvalue_update2(:,3);

load('data_qvalue_update3.mat')
data_qvalue_update3  = qvalueupdate3;

update3 = data_qvalue_update3(:,1);
qvalue3 = data_qvalue_update3(:,2);
gvalue3 = data_qvalue_update3(:,3);

figure(1);
plot(update1, qvalue1, 'b-', 'LineWidth', 2);
hold on;
plot(update2, qvalue2, 'g-', 'LineWidth', 2);
plot(update3, qvalue3, 'm-', 'LineWidth', 2);
hold off;
xlabel('Update Number'); 
ylabel('Q Value'); 
legend('\alpha = 0.1', '\alpha = 0.01', '\alpha = 0.001')
title('Q Learning Value Over Update Number');
grid on;

% saveas(gcf, fullfile(folder_path, 'q_alpha_qvalue_update_analysis.jpg'));
