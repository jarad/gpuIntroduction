load 'results/real_pop_00001.txt'
load 'results/real_pop_00002.txt'
load 'results/real_pop_00004.txt'
load 'results/real_pop_00008.txt'
load 'results/real_pop_00032.txt'
load 'results/real_pop_00128.txt'
load 'results/real_pop_00512.txt'
load 'results/real_pop_02048.txt'
load 'results/real_pop_08192.txt'
load 'results/real_pop_32768.txt'

% load 'results2/real_pop_00001.txt'
% load 'results2/real_pop_00002.txt'
% load 'results2/real_pop_00004.txt'
% load 'results2/real_pop_00008.txt'
% load 'results2/real_pop_00032.txt'
% load 'results2/real_pop_00128.txt'
% load 'results2/real_pop_00512.txt'
% load 'results2/real_pop_02048.txt'
% load 'results2/real_pop_08192.txt'
% load 'results2/real_pop_32768.txt'

N = size(real_pop_00008,1);
b = 8192 + 1;
h = 1;

counts1 = binModes(real_pop_00001(b:N,:),h);
counts2 = binModes(real_pop_00002(b:N,:),h);
counts4 = binModes(real_pop_00004(b:N,:),h);
counts8 = binModes(real_pop_00008(b:N,:),h);
counts32 = binModes(real_pop_00032(b:N,:),h);
counts128 = binModes(real_pop_00128(b:N,:),h);
counts512 = binModes(real_pop_00512(b:N,:),h);
counts2048 = binModes(real_pop_02048(b:N,:),h);
counts8192 = binModes(real_pop_08192(b:N,:),h);
counts32768 = binModes(real_pop_32768(b:N,:),h);

t1 = timetovisitallmodes(iteration_modes(real_pop_00001(b:N,:),1),24);
t2 = timetovisitallmodes(iteration_modes(real_pop_00002(b:N,:),1),24);
t4 = timetovisitallmodes(iteration_modes(real_pop_00004(b:N,:),1),24);
t8 = timetovisitallmodes(iteration_modes(real_pop_00008(b:N,:),1),24);
t32 = timetovisitallmodes(iteration_modes(real_pop_00032(b:N,:),1),24);
t128 = timetovisitallmodes(iteration_modes(real_pop_00128(b:N,:),1),24);
t512 = timetovisitallmodes(iteration_modes(real_pop_00512(b:N,:),1),24);
t2048 = timetovisitallmodes(iteration_modes(real_pop_02048(b:N,:),1),24);
t8192 = timetovisitallmodes(iteration_modes(real_pop_08192(b:N,:),1),24);
t32768 = timetovisitallmodes(iteration_modes(real_pop_32768(b:N,:),1),24);


figure(1);
subplot(2,5,1);
bar(counts1, 1);
set(gca,'XTick',[]);
xlabel('M=1');
axis tight;
subplot(2,5,2);
bar(counts2, 1);
set(gca,'XTick',[]);
xlabel('M=2');
axis tight;
subplot(2,5,3);
bar(counts4, 1);
set(gca,'XTick',[]);
xlabel('M=4');
axis tight;
subplot(2,5,4);
bar(counts8, 1);
set(gca,'XTick',[]);
xlabel('M=8');
axis tight;
subplot(2,5,5);
bar(counts32, 1);
set(gca,'XTick',[]);
xlabel('M=32');
axis tight;
subplot(2,5,6);
bar(counts128, 1);
set(gca,'XTick',[]);
xlabel('M=128');
axis tight;
subplot(2,5,7);
bar(counts512, 1);
set(gca,'XTick',[]);
xlabel('M=512');
axis tight;
subplot(2,5,8);
bar(counts2048, 1);
set(gca,'XTick',[]);
xlabel('M=2048');
axis tight;
subplot(2,5,9);
bar(counts8192, 1);
set(gca,'XTick',[]);
xlabel('M=8192');
axis tight;
subplot(2,5,10);
bar(counts32768, 1);
set(gca,'XTick',[]);
xlabel('M=32768');
axis tight;
print -f1 -depsc -tiff -r300 bar_modes_mcmc

xs = [1 2 4 8 32 128 512 2048 8192 32768];
ys = [mean(t1) mean(t2) mean(t4) mean(t8) mean(t32) mean(t128) mean(t512) mean(t2048) mean(t8192) mean(t32768)];

figure(2)
bar(ys)
set(gca,'Xtick',1:10,'xticklabel',xs)
xlabel('M');
ylabel('Iterations');
print -f2 -depsc -tiff -r300 iterations_traverse

% figure(3)
[xb,yb,bins] = binup(real_pop_00128(b:N,1),real_pop_00128(b:N,2),-4,7,-4,7,100);
% contour(xb,yb,bins);
% xlabel(['\mu_' int2str(1)])
% ylabel(['\mu_' int2str(2)])

figure(3)
surf(xb,yb,bins);
shading interp;
colormap(jet);
xlabel(['\mu_' int2str(1)])
ylabel(['\mu_' int2str(2)])
zlabel('Empirical Density')
set(gca,'ZTick',[]);
view([-33 38]);
axis tight;
print -f3 -depsc -tiff -r150 -zbuffer modes_12_mcmc