load 'results/xs_mvlg.txt'
load 'results/ws_mvlg.txt'
load 'results/xs_real_mvlg.txt'
load 'results/ys_real_mvlg.txt'

T = 200;
N = size(ws_mvlg,1);
Dx = size(xs_real_mvlg,1) / T;
Dy = size(ys_real_mvlg,1) / T;

xsmvlg = reshape(xs_mvlg,Dx,N,T);
muxsmvlg = zeros(T,Dx);
mvlgxsr = reshape(xs_real_mvlg,Dx,T)';
sdxsmvlg = zeros(T,Dx);
sumws = sum(ws_mvlg);

for i=1:T
    for j=1:Dx
        vals = reshape(xsmvlg(j,:,i),N,1);
        muxsmvlg(i,j) = sum(vals.*ws_mvlg)/sumws;
        sdxsmvlg(i,j) = sqrt(sum(vals.*vals.*ws_mvlg)/sumws - muxsmvlg(i,j)^2);
    end
end


for j=1:Dx
    fig = figure(j);
    set(fig,'position',[100 100 1000 400]);
    hold on;
    plot(mvlgxsr(:,j),'+-b');
    plot(muxsmvlg(:,j),'r');
    plot(muxsmvlg(:,j) + sdxsmvlg(:,j), 'k--', 'color', [1 .6 .6]);
    plot(muxsmvlg(:,j) - sdxsmvlg(:,j), 'k--', 'color', [1 .6 .6]);
    h = legend('real','filter mean','+/- 1 S.D.','location','NorthWest');
    xlabel('Time');
    str = ['x_' int2str(j)];
    ylabel(str);
%     axis image;
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0.25 2.5 16.0 6.0]);
end

% print -f1 -depsc -tiff -r300 fsv_alpha1
% print -f2 -depsc -tiff -r300 fsv_alpha2
% print -f3 -depsc -tiff -r300 fsv_alpha3