clear all;
close all;
clc;

alpha = 0.4;
gamma = 0.99;
r = -1;
Q_initial = -200;
Q_next = 0;
Q = -197.233;
Q = Q + alpha * ((r + gamma * Q_next) - Q);