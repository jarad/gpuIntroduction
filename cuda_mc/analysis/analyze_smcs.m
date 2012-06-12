load 'results/smcs_mgmu_forget_001024.txt'
load 'results/smcs_mgmu_forget_002048.txt'
load 'results/smcs_mgmu_forget_004096.txt'
load 'results/smcs_mgmu_forget_008192.txt'
load 'results/smcs_mgmu_forget_016384.txt'
load 'results/smcs_mgmu_forget_032768.txt'
load 'results/smcs_mgmu_forget_065536.txt'
load 'results/smcs_mgmu_forget_131072.txt'
load 'results/smcs_mgmu_forget_w_001024.txt'
load 'results/smcs_mgmu_forget_w_002048.txt'
load 'results/smcs_mgmu_forget_w_004096.txt'
load 'results/smcs_mgmu_forget_w_008192.txt'
load 'results/smcs_mgmu_forget_w_016384.txt'
load 'results/smcs_mgmu_forget_w_032768.txt'
load 'results/smcs_mgmu_forget_w_065536.txt'
load 'results/smcs_mgmu_forget_w_131072.txt'

smcs_mgmu_forget_w_001024 = norm_weights(smcs_mgmu_forget_w_001024);
smcs_mgmu_forget_w_002048 = norm_weights(smcs_mgmu_forget_w_002048);
smcs_mgmu_forget_w_004096 = norm_weights(smcs_mgmu_forget_w_004096);
smcs_mgmu_forget_w_008192 = norm_weights(smcs_mgmu_forget_w_008192);
smcs_mgmu_forget_w_016384 = norm_weights(smcs_mgmu_forget_w_016384);
smcs_mgmu_forget_w_032768 = norm_weights(smcs_mgmu_forget_w_032768);
smcs_mgmu_forget_w_065536 = norm_weights(smcs_mgmu_forget_w_065536);
smcs_mgmu_forget_w_131072 = norm_weights(smcs_mgmu_forget_w_131072);

%N = size(smcs_mgmu_forget_008192,1);
h = 1;

counts1024 = binModes_w(smcs_mgmu_forget_001024,smcs_mgmu_forget_w_001024,h);
counts2048 = binModes_w(smcs_mgmu_forget_002048,smcs_mgmu_forget_w_002048,h);
counts4096 = binModes_w(smcs_mgmu_forget_004096,smcs_mgmu_forget_w_004096,h);
counts8192 = binModes_w(smcs_mgmu_forget_008192,smcs_mgmu_forget_w_008192,h);
counts16384 = binModes_w(smcs_mgmu_forget_016384,smcs_mgmu_forget_w_016384,h);
counts32768 = binModes_w(smcs_mgmu_forget_032768,smcs_mgmu_forget_w_032768,h);
counts65536 = binModes_w(smcs_mgmu_forget_065536,smcs_mgmu_forget_w_065536,h);
counts131072 = binModes_w(smcs_mgmu_forget_131072,smcs_mgmu_forget_w_131072,h);

figure(1);
subplot(2,4,1);
bar(counts1024,1);
set(gca,'XTick',[]);
xlabel('N=1024');
axis tight;
subplot(2,4,2);
bar(counts2048,1);
set(gca,'XTick',[]);
xlabel('N=2048');
axis tight;
subplot(2,4,3);
bar(counts4096,1);
set(gca,'XTick',[]);
xlabel('N=4096');
axis tight;
subplot(2,4,4);
bar(counts8192,1);
set(gca,'XTick',[]);
xlabel('N=8192');
axis tight;
subplot(2,4,5);
bar(counts16384,1);
set(gca,'XTick',[]);
xlabel('N=16384');
axis tight;
subplot(2,4,6);
bar(counts32768,1);
set(gca,'XTick',[]);
xlabel('N=32768');
axis tight;
subplot(2,4,7);
bar(counts65536,1);
set(gca,'XTick',[]);
xlabel('N=65536');
axis tight;
subplot(2,4,8);
bar(counts131072,1);
set(gca,'XTick',[]);
xlabel('N=131072');
axis tight;
print -f1 -depsc -tiff -r300 bar_modes_smcs

figure(2)
[xb,yb,bins] = binup_w(smcs_mgmu_forget_131072(:,1),smcs_mgmu_forget_131072(:,2),smcs_mgmu_forget_w_131072,-4,7,-4,7,100);
surf(xb,yb,bins);
shading interp;
colormap(jet);
xlabel(['\mu_' int2str(1)])
ylabel(['\mu_' int2str(2)])
zlabel('Empirical Density')
set(gca,'ZTick',[]);
view([-33 38]);
axis tight;
print -f2 -depsc -tiff -r150 -zbuffer modes_12_smcs
