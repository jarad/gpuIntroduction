load 'results/smcs_mgmu.txt'
load 'results/smcs_mgmu_forget_065536.txt'
load 'results/smcs_mgmu_forget_w_065536.txt'

T = 200;
N = 65536;
D = 4;

t1 = 1;
t2 = 26;
t3 = 62;
t4 = 112;
t5 = 173;

data = reshape(smcs_mgmu',D,N,T);

data1 = data(:,:,t1);
data2 = data(:,:,t2);
data3 = data(:,:,t3);
data4 = data(:,:,t4);
data5 = data(:,:,t5);

data1 = data1';
data2 = data2';
data3 = data3';
data4 = data4';
data5 = data5';

w = ones(N,1);


[xb1,yb1,bins1] = binup_w(data1(:,1),data1(:,2),w,-10.5,10.5,-10.5,10.5,100);
[xb2,yb2,bins2] = binup_w(data2(:,1),data2(:,2),w,-10.5,10.5,-10.5,10.5,100);
[xb3,yb3,bins3] = binup_w(data3(:,1),data3(:,2),w,-10.5,10.5,-10.5,10.5,100);
[xb4,yb4,bins4] = binup_w(data4(:,1),data4(:,2),w,-10.5,10.5,-10.5,10.5,100);
[xb5,yb5,bins5] = binup_w(data5(:,1),data5(:,2),w,-10.5,10.5,-10.5,10.5,100);

figure(1)
surf(xb1,yb1,bins1);
shading interp;
colormap(jet);
xlabel(['\mu_' int2str(1)])
ylabel(['\mu_' int2str(2)])
zlabel('Empirical Density')
set(gca,'ZTick',[]);
view([-33 38]);
axis tight;
print -f1 -depsc -tiff -r150 -zbuffer modes_12_smcs_t1

figure(2)
surf(xb2,yb2,bins2);
shading interp;
colormap(jet);
xlabel(['\mu_' int2str(1)])
ylabel(['\mu_' int2str(2)])
zlabel('Empirical Density')
set(gca,'ZTick',[]);
view([-33 38]);
axis tight;
print -f2 -depsc -tiff -r150 -zbuffer modes_12_smcs_t2

figure(3)
surf(xb3,yb3,bins3);
shading interp;
colormap(jet);
xlabel(['\mu_' int2str(1)])
ylabel(['\mu_' int2str(2)])
zlabel('Empirical Density')
set(gca,'ZTick',[]);
view([-33 38]);
axis tight;
print -f3 -depsc -tiff -r150 -zbuffer modes_12_smcs_t3

figure(4)
surf(xb4,yb4,bins4);
shading interp;
colormap(jet);
xlabel(['\mu_' int2str(1)])
ylabel(['\mu_' int2str(2)])
zlabel('Empirical Density')
set(gca,'ZTick',[]);
view([-33 38]);
axis tight;
print -f4 -depsc -tiff -r150 -zbuffer modes_12_smcs_t4

figure(5)
surf(xb5,yb5,bins5);
shading interp;
colormap(jet);
xlabel(['\mu_' int2str(1)])
ylabel(['\mu_' int2str(2)])
zlabel('Empirical Density')
set(gca,'ZTick',[]);
view([-33 38]);
axis tight;
print -f5 -depsc -tiff -r150 -zbuffer modes_12_smcs_t5

smcs_mgmu_forget_w_065536 = norm_weights(smcs_mgmu_forget_w_065536);
[xbT,ybT,binsT] = binup_w(smcs_mgmu_forget_065536(:,1),smcs_mgmu_forget_065536(:,2),smcs_mgmu_forget_w_065536,-10.5,10.5,-10.5,10.5,100);

figure(6)
surf(xbT,ybT,binsT);
shading interp;
colormap(jet);
xlabel(['\mu_' int2str(1)])
ylabel(['\mu_' int2str(2)])
zlabel('Empirical Density')
set(gca,'ZTick',[]);
view([-33 38]);
axis tight;
print -f6 -depsc -tiff -r150 -zbuffer modes_12_smcs_t6

% bins0 = 100*ones(100,100) + randn(100,100);
bins0 = ones(100,100);
for i=1:100
    for j=1:100
        if xbT(i) < -10 || xbT(i) > 10 ||  ybT(j) < -10 || ybT(j) > 10
            bins0(i,j) = 0;
        end
    end
end
figure(7)
surf(xbT,ybT,bins0);
shading interp;
colormap(jet);
xlabel(['\mu_' int2str(1)])
ylabel(['\mu_' int2str(2)])
zlabel('Empirical Density')
set(gca,'ZTick',[]);
view([-33 38]);
axis tight;
print -f7 -depsc -tiff -r150 -zbuffer modes_12_smcs_t0