load 'results/xs_fsv.txt'
load 'results/ws_fsv.txt'
load 'results/fsv_xs_real.txt'
load 'results/fsv_ys_real.txt'

T = 200;
N = size(ws_fsv,1);
Dx = size(fsv_xs_real,1) / T;
Dy = size(fsv_ys_real,1) / T;

xsfsv = reshape(xs_fsv,Dx,N,T);
muxsfsv = zeros(T,Dx);
fsvxsr = reshape(fsv_xs_real,Dx,T)';
sdxsfsv = zeros(T,Dx);
sumws = sum(ws_fsv);

for i=1:T
    for j=1:Dx
        vals = reshape(xsfsv(j,:,i),N,1);
        muxsfsv(i,j) = sum(vals.*ws_fsv)/sumws;
        sdxsfsv(i,j) = sqrt(sum(vals.*vals.*ws_fsv)/sumws - muxsfsv(i,j)^2);
    end
end


for j=1:Dx
    fig = figure(j);
    set(fig,'position',[100 100 1000 400]);
    hold on;
    axis([0 200 -4 5])
    plot(fsvxsr(:,j),'+-b');
    plot(muxsfsv(:,j),'r');
    plot(muxsfsv(:,j) + sdxsfsv(:,j), 'k--', 'color', [1 .6 .6]);
    plot(muxsfsv(:,j) - sdxsfsv(:,j), 'k--', 'color', [1 .6 .6]);
    h = legend('real','filter mean','+/- 1 S.D.','location','NorthWest');
    xlabel('Time');
    str = ['x_' int2str(j)];
    ylabel(str);

    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0.25 0.25 18.0 12.0]);
%     set(gcf, 'PaperOrientation', 'landscape')
end

print -f1 -depsc -tiff -r300 fsv_alpha1
print -f2 -depsc -tiff -r300 fsv_alpha2
print -f3 -depsc -tiff -r300 fsv_alpha3