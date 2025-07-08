clear all;
close all;
clc;

load('data_bestactions.mat')
bestaction = bestactions;

best_weight = bestaction(:, 1);
best_action = bestaction(:, 2);

figure(1);
scatter(best_weight(best_action == 1), best_action(best_action == 1), 40, [0, 0, 1], 'o', 'filled', 'DisplayName', 'Coarse Feed (1)');
hold on;
scatter(best_weight(best_action == -1), best_action(best_action == -1), 40, [1, 0, 0], '^', 'filled', 'DisplayName', 'Fine Feed (-1)');
title('Sigmoid Function Fit of Best Action vs Weight');
xlabel('Weight');
ylabel('Action');
legend('Location', 'southwest', 'FontSize', 6);
grid on;

% Define parameters
a = 2.007;
b = 0.7;
c = 44.5;
d = 1.003;

% Generate x values
x = linspace(min(best_weight), max(best_weight), 1000);

% Calculate corresponding y values using the sigmoid function
y = -a./(1 + exp(-b*(x - c))) + d;

% Plot the sigmoid function
plot(x, y, 'LineWidth', 2);

% Find the transition field
x_last_1 = x(find(y >= 1, 1, 'last'));
x_first_minus_1 = x(find(y <= -1, 1, 'first'));

% Plot vertical lines
plot([x_last_1, x_last_1], ylim, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Transition start');
plot([x_first_minus_1, x_first_minus_1], ylim, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Transition end');

% Find the x-coordinate where y = 0, y = 0.8 and y = -0.8
x_0 = interp1(y, x, 0);
x_08 = interp1(y, x, 0.8);
x_minus_08 = interp1(y, x, -0.8);

% Plot the point where y = 0, y = 0.8 and y = -0.8
scatter(x_0, 0, 100, 'k', 'x', 'LineWidth', 2, 'DisplayName', ' P(coarse) = 0.5 & P(fine) = 0.5');
scatter(x_08, 0.8, 100, 'g', 'x', 'LineWidth', 2, 'DisplayName', 'P(coarse) = 0.8 & P(fine) = 0.2');
scatter(x_minus_08, -0.8, 100, 'm', 'x', 'LineWidth', 2, 'DisplayName', 'P(coarse) = 0.2 & P(fine) = 0.8');

hold off;