clear;close all;clc;

monte_carlo = ones(1, 76); 
td = [ones(1, 52), -ones(1, 76-52)]; 
q_learning = [ones(1, 51), -ones(1, 76-51)];

x = 1:75;

figure;
hold on;

% Plot Monte Carlo
plot(x, monte_carlo, 'bo', 'DisplayName', 'Monte Carlo');

% Plot TD
plot(x, td, 'r^', 'DisplayName', 'TD');

% Plot Q-Learning
plot(x, q_learning, 'gs', 'DisplayName', 'Q-Learning');

% Add titles and labels
title('Best Action vs State/Weight');
xlabel('State/Weight');
ylabel('Action');

% Add legend
legend('Location', 'best');

% Add grid
grid on;

% Display the plot
hold off;
