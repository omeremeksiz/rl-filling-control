clear all
close all
clc

load('qValueOutput.mat')
data  = qValueoutput;
s = data(:,1:2); %Time and Weight
action = data(:,3);
qValue = data(:,4);

s1 = (1.2:0.01:7.5); %Time
s2 = (1:80); %Weight 10K quantized

[S1,S2] = meshgrid(s1,s2);

for c = 1:numel(s1)
    for k = 1:numel(s2)
        
        s_1 = s1(c);
        s_2 = s2(k);
               
        [val,ia,ib]=intersect(s,[s_1 s_2],'rows');    
        if ia
            A(k,c) = action(ia);
            Qval(k,c) = qValue(ia);
        else
            A(k,c) = 0;
            Qval(k,c) = 0;
        end
    end
end

figure(1)
meshc(10*S1,20*S2,A)
xlabel('Time Step - milliseconds')
ylabel('Weight - grams')
zlabel('Action')
grid on
view(2)
colorbar

figure(2)
surf(10*S1,20*S2,-10*log10(-Qval))
xlabel('Time Step - milliseconds')
ylabel('Weight - grams')
zlabel('Q Value in (- dB)')
grid on
colorbar


