clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/app_out/comp_app_out/a_01';

load('mc_qvalue_update.mat')
mc_qvalue_update  = qvalueupdate1;

load('td_qvalue_update.mat')
td_qvalue_update  = qvalueupdate2;

load('ql_qvalue_update.mat')
ql_qvalue_update  = qvalueupdate3;

mc_update = mc_qvalue_update(:,1);
mc_qValue = mc_qvalue_update(:,2);

td_update = td_qvalue_update(:, 1);
td_qValue = td_qvalue_update(:, 2);

ql_update = ql_qvalue_update(:, 1);
ql_qValue = ql_qvalue_update(:, 2);

figure;
plot(mc_update, mc_qValue, 'b-', 'LineWidth', 4);
hold on;
plot(td_update, td_qValue, 'r-', 'LineWidth', 4); 
plot(ql_update, ql_qValue, 'm-', 'LineWidth', 4); 
hold off;
xlabel('Güncelleme Sayısı','FontSize', 20, 'FontWeight', 'bold');
ylabel('Q Değeri','FontSize', 20, 'FontWeight', 'bold');
legend('MC', 'TD', 'QL', 'FontSize', 20, 'Location', 'northwest');
set(gca,'FontSize', 16, 'FontWeight', 'bold');
grid on;

% saveas(gcf, fullfile(folder_path, 'comp_qvalue_update_analysis_53_minus_1.jpg'));
