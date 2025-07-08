clear all;
close all;
clc;

load('data_qValue.mat')
data_qValue  = qValueoutput;

weight = data_qValue(:,1);
weight_diff = data_qValue(:,2);
action = data_qValue(:,3);
qValue = data_qValue(:,4);

% Reward vs (Weight, Weight Difference)

unique_weights = unique(weight);
unique_diffs = unique(weight_diff);

[X, Y] = meshgrid(unique_weights, unique_diffs);

Z = griddata(weight, weight_diff, qValue, X, Y, 'cubic');

figure(1)
surf(X, Y, Z, 'FaceColor', 'interp', 'EdgeColor', 'none');
title('Q Value vs (Weight, Weight Difference)');
xlabel('Weight');
ylabel('Weight Difference');
zlabel('Q Value');
ylim([0 10])
grid on;
colorbar;

view(0,90);

% Action vs Weight

unique_weights = unique(weight);
unique_diffs = unique(weight_diff);

[X, Y] = meshgrid(unique_weights, unique_diffs);

Z = griddata(weight, weight_diff, action, X, Y, 'cubic');

figure(2)
mesh(X, Y, Z, 'FaceColor', 'interp', 'EdgeColor', 'none');
title('Action vs (Weight, Weight Difference)');
xlabel('Weight');
ylabel('Weight Difference');
zlabel('Action');
ylim([0 10])
zlim([-1 1]);
grid on;
h = colorbar;
h.Limits = [-1 1];

% view(0,90);

% Update Count vs Weight

load('data_countlist.mat')
data_countlist  = countlist;

weight = data_countlist(:,1);
weight_diff = data_countlist(:,2);
action = data_countlist(:,3);
count = data_countlist(:,4);

unique_weights = unique(weight);
unique_weight_diff = unique(weight_diff);
unique_actions = [1, -1];

count_matrix = zeros(length(unique_weights), length(unique_weight_diff), length(unique_actions));

for i = 1:length(weight)
    w_idx = find(unique_weights == weight(i));
    wd_idx = find(unique_weight_diff == weight_diff(i));
    a_idx = find(unique_actions == action(i));
    
    count_matrix(w_idx, wd_idx, a_idx) = count(i);
end

fig_counter = 3;

for a_idx = 1:length(unique_actions)
    figure(fig_counter)
    
    current_action_matrix = squeeze(count_matrix(:, :, a_idx));
    
    bar3(current_action_matrix);
    
    xlabel('Weight Difference');
    ylabel('Weight');
    zlabel('Count');

    if a_idx == 1
        title(['Count vs (Weight, Weight Difference) for Coarse Feed (' num2str(unique_actions(a_idx)) ')']);
    else
        title(['Count vs (Weight, Weight Difference) for Fine Feed (' num2str(unique_actions(a_idx)) ')']);
    end

    fig_counter = fig_counter + 1;
end

% Best Actions and Weight
load('data_bestActions.mat')
bestAction = bestactions;

weight = bestAction(:, 1);
weight_diff = bestAction(:,2);
action = bestAction(:, 3);

unique_weights = unique(weight);
unique_diffs = unique(weight_diff);

[X, Y] = meshgrid(unique_weights, unique_diffs);

Z = griddata(weight, weight_diff, action, X, Y, 'cubic');

markerSize = 150;
markerColor = 'b';

figure(5)
surf(X, Y, Z, 'FaceColor', 'interp', 'EdgeColor', 'none');
title('Best Action vs (Weight, Weight Difference)')
xlabel('Weight');
ylabel('Weight Difference');
zlabel('Action');
ylim([0 10])
zlim([-1 1]);
grid on;
h = colorbar;
h.Limits = [-1 1];

view(0, 90);