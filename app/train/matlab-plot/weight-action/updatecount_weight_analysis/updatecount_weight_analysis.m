clear all;
close all;
clc;

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/monte_carlo/RL_imp_v6/application_output_v3/updatecount_weight_analysis';

load('data_qvalue.mat')
data_qvalue  = qvalueoutput;

count_weight = data_qvalue(:, 1);
count_action = data_qvalue(:, 2);
count = data_qvalue(:, 3);

%% Update Count vs Weight

unique_weights = unique(count_weight);

figure(1)

for i = 1:length(count_weight)
    if count_action(i) == 1
        barColor = 'b'; % Blue for action = 1
    else
        barColor = 'r'; % Red for action = -1
    end
    
    bar(count_weight(i), count(i), 'FaceColor', barColor, 'EdgeColor', 'k');
    
    hold on;
end

colormap([1 0 0; 0 0 1]);

title('Update Count vs Weight');
xlabel('Weight');
ylabel('Count');
grid on;

xticks(0:5:max(unique_weights));

c = colorbar;
c.Label.String = 'Action Type';
c.Ticks = [0, 1];  % Set ticks for -1 and 1
c.TickLabels = {'Action -1', 'Action 1'};

hold off;

% saveas(gcf, fullfile(folder_path, 'updatecount_weight_analysis.jpg'));