clear all;
close all;
clc;

% Penalty Analysis: 
% Case - 1: Overflow = 0 Underflow = 0
% Case - 2: Overflow = -30 Underflow = -15
% Case - 3: Overflow = -50 Underflow = -30
% Case - 4: Overflow = -100 Underflow = -50

folder_path = '/Users/omeremeksiz/Desktop/masaustu/RL/application/td_learning/application_output_v2/penalty_analysis';

load('data_bestactions1.mat')
bestaction_1 = bestactions1;

weight_1 = bestaction_1(:,1);
qValue_1 = bestaction_1(:,4);

load('data_bestactions2.mat')
bestaction_2 = bestactions2;

qValue_2 = bestaction_2(:,4);

load('data_bestactions3.mat')
bestaction_3 = bestactions3;

qValue_3 = bestaction_3(:,4);

load('data_bestactions4.mat')
bestaction_4 = bestactions4;

qValue_4 = bestaction_4(:,4);

% Q Value vs Weight

figure(1)
plot(weight_1, qValue_1, 'LineWidth', 2);
title('Q Value vs Weight');
xlabel('Weight');
ylabel('Q Value');
grid on;
hold on

plot(weight_1, qValue_2, 'LineWidth', 2);
plot(weight_1, qValue_3, 'LineWidth', 2);
plot(weight_1, qValue_4, 'LineWidth', 2);

legend('Case - 1: $C_1$ = 0 $C_2$ = 0, SS: 71' ...
    , 'Case - 2: $C_1$ = -30 $C_2$ = -15, SS: 53' ...
    , 'Case - 3: $C_1$ = -50 $C_2$ = -30, SS: 49' ...
    , 'Case - 4: $C_1$ = -100 $C_2$ = -50, SS: 47', 'Interpreter', 'latex','Location','northwest');

legendFontSize = 12;
legend('FontSize', legendFontSize);

hold off;

saveas(gcf, fullfile(folder_path, 'penalty_analysis.jpg'));