clear; close all; clc;
% Default plot
% monte_carlo = ones(1, 76); 
% td = [ones(1, 52), -ones(1, 76-52)];
% q_learning = [ones(1, 51), -ones(1, 76-51)];
% 
% x = 1:76;
% 
% figure();
% 
% scatter(x, monte_carlo, 90, [0, 0, 1], 'o', 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Monte Carlo');
% hold on;
% scatter(x, td, 90, [0, 0.5, 0], '^', 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'TD');
% scatter(x, q_learning, 90, [1, 0, 0], 's', 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Q-Learning');
% 
% title('Best Action vs State/Weight');
% xlabel('State');
% ylabel('Action');
% ylim([-1 1]); 
% legend('show', 'Location', 'northeast');
% grid on;
% hold off;

% Heatmap
% monte_carlo = ones(1, 76); 
% td = [ones(1, 52), -ones(1, 76-52)];
% q_learning = [ones(1, 51), -ones(1, 76-51)];
% 
% data = [monte_carlo; td; q_learning];
% x = 1:76;
% methods = {'Monte Carlo', 'TD', 'QL'};
% 
% figure();
% imagesc(x, 1:3, data);
% colormap([1 0 0; 0 0 1]); % Red for -1, Blue for 1
% caxis([-1 1]);
% colorbar('Ticks', [-1, 1], 'TickLabels', {'-1', '1'});
% 
% yticks(1:3);
% yticklabels(methods);
% xticks(0:10:80);
% xlabel('State');
% ylabel('Method');
% title('Best Action vs State');
% 
% hold on;
% for i = 1.5:1:2.5
%     yline(i, 'k--', 'LineWidth', 1.2);
% end
% 
% for i = 5:5:75
%     xline(i, 'k--', 'LineWidth', 1.2);
% end
% 
% for i = 1:3
%     idx = find(data(i, :) == -1, 1);
%     if ~isempty(idx)
%         plot(idx, i, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'yellow');
%         text(idx, i + 0.1, sprintf('%s: State %d', methods{i}, idx), ...
%             'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
%             'Color', 'white', 'FontSize', 10);
%     end
% end
% hold off;

% % Subplots mc vs td vs ql
% monte_carlo = ones(1, 76); 
% td = [ones(1, 52), -ones(1, 76-52)];
% q_learning = [ones(1, 51), -ones(1, 76-51)];
% 
% x = 1:76;
% methods = {'Monte Carlo', 'TD', 'Q-Learning'};
% data = {monte_carlo, td, q_learning};
% colors = {[0 0 1], [0 0 1], [0 0 1]}; 
% 
% figure();
% for i = 1:3
%     subplot(3, 1, i);
%     hold on;
% 
%     ones_idx = find(data{i} == 1);
%     neg_ones_idx = find(data{i} == -1);
% 
%     scatter(x(ones_idx), data{i}(ones_idx), 60, colors{i}, 'o', 'filled', 'DisplayName', 'Kaba Besleme (1)');
%     scatter(x(neg_ones_idx), data{i}(neg_ones_idx), 60, [1 0 0], '^', 'filled', 'DisplayName', 'İnce Besleme (-1)');
% 
%     title(methods{i});
%     ylim([-1 1]);
%     grid on;
% 
%     legend('Location', 'northeast'); 
% 
%     hold off;
% end
% 
% xlabel('Durum');

monte_carlo = ones(1, 76); 
monte_carlo_cezali = [ones(1, 46), -ones(1, 76-46)]; 

x = 1:76;
methods = {'Monte Carlo (Cezasız)', 'Monte Carlo (Cezalı)'};
data = {monte_carlo, monte_carlo_cezali};
colors = {[0 0 1], [0 0 1]};  

figure();
subplot_handles = gobjects(2,1); 

for i = 1:2
    subplot_handles(i) = subplot(2, 1, i); 
    hold on;

    ones_idx = find(data{i} == 1);
    neg_ones_idx = find(data{i} == -1);

    scatter(x(ones_idx), data{i}(ones_idx), 60, colors{i}, 'o', 'filled', 'DisplayName', 'Kaba Besleme (1)');
    scatter(x(neg_ones_idx), data{i}(neg_ones_idx), 60, [1 0 0], '^', 'filled', 'DisplayName', 'İnce Besleme (-1)');

    title(methods{i}, 'FontSize', 14, 'FontWeight', 'bold');
    ylim([-1 1]);
    yticks([-1 0 1]); 
    grid on;

    legend('Location', 'northeast', 'FontSize', 12, 'FontWeight', 'bold');

    ax = gca; 
    ax.FontSize = 14;
    ax.FontWeight = 'bold'; 

    hold off;
end

annotation('textbox', [0.52, 0.03, 0, 0], 'String', 'Durum', 'FontSize', 16, 'FontWeight', 'bold', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'EdgeColor', 'none');

annotation('textbox', [0.05, 0.52, 0, 0], 'String', 'Eylem', 'FontSize', 16, 'FontWeight', 'bold', ...
           'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Rotation', 90, 'EdgeColor', 'none');
