clear all;
close all;
clc;

load('data_qValue_count0.mat')
data_qValue_count0  = qValueoutput1;

weight_count0 = data_qValue_count0(:,1);
weight_diff_count0 = data_qValue_count0(:,2);
action_count0 = data_qValue_count0(:,3);
qValue_count0 = data_qValue_count0(:,4);

% Reward vs (Weight, Weight Difference)

unique_weights = unique(weight_count0);
unique_diffs = unique(weight_diff_count0);

[X, Y] = meshgrid(unique_weights, unique_diffs);

Z = griddata(weight_count0, weight_diff_count0, qValue_count0, X, Y, 'cubic');

figure(1)
surf(X, Y, Z, 'FaceColor', 'interp', 'EdgeColor', 'none');
title('Q Value vs (Weight, Weight Difference)');
xlabel('Weight');
ylabel('Weight Difference');
zlabel('Q Value');
grid on;
hold on;

load('data_qValue_count30.mat')
data_qValue_count30  = qValueoutput2;

weight_count30 = data_qValue_count30(:,1);
weight_diff_count30 = data_qValue_count30(:,2);
action_count30 = data_qValue_count30(:,3);
qValue_count30 = data_qValue_count30(:,4);

% Reward vs (Weight, Weight Difference)

unique_weights = unique(weight_count30);
unique_diffs = unique(weight_diff_count30);

[X, Y] = meshgrid(unique_weights, unique_diffs);

Z = griddata(weight_count30, weight_diff_count30, qValue_count30, X, Y, 'cubic');

surf(X, Y, Z, 'FaceColor', 'interp', 'EdgeColor', 'none');

load('data_qValue_count50.mat')
data_qValue_count50  = qValueoutput3;

weight_count50 = data_qValue_count50(:,1);
weight_diff_count50 = data_qValue_count50(:,2);
action_count50 = data_qValue_count50(:,3);
qValue_count50 = data_qValue_count50(:,4);

% Reward vs (Weight, Weight Difference)

unique_weights = unique(weight_count50);
unique_diffs = unique(weight_diff_count50);

[X, Y] = meshgrid(unique_weights, unique_diffs);

Z = griddata(weight_count50, weight_diff_count50, qValue_count50, X, Y, 'cubic');

surf(X, Y, Z, 'FaceColor', 'interp', 'EdgeColor', 'none');

legend('Filtered < 0 Update Count', 'Filtered < 30 Update Count', 'Filtered < 50 Update Count', 'Location','northwest');

legendFontSize = 12;
legend('FontSize', legendFontSize);

hold off;