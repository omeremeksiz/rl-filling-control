clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/monte_carlo/RL_imp_v6/application_output_v2/tolerance_limit';

load('data_tolerance_pairs.mat')
data_tolerance_pairs = tolerancepairs;

cutoff_weight = data_tolerance_pairs(:, 1);
error_type = data_tolerance_pairs(:, 2);
weight_margin = data_tolerance_pairs(:, 3);

unique_cutoff_weight = unique(cutoff_weight);

red_indices = find(error_type == 1001);
blue_indices = find(error_type == -1001);

figure(1);
bar(cutoff_weight(red_indices), weight_margin(red_indices), 'r');
xlabel('Cutoff Weight');
ylabel('Weight Margin');
title('Overflow - Weight Margin vs Cutoff Weight');
grid on;
xticks(unique_cutoff_weight);
xlim([46 76]);
ylim([0 15]);
yticks(0:15);

saveas(gcf, fullfile(folder_path, 'overflow.jpg'));

figure(2);
bar(cutoff_weight(blue_indices), weight_margin(blue_indices), 'b');
xlabel('Cutoff Weight');
ylabel('Weight Margin');
title('Underflow - Weight Margin vs Cutoff Weight');
grid on;
xticks(unique_cutoff_weight);
xlim([46 76]);
ylim([0 15]);
yticks(0:15);

saveas(gcf, fullfile(folder_path, 'underflow.jpg'));