function [x,y,bins] = binup_w(data1,data2,ws,xmin,xmax,ymin,ymax,N)
bins = zeros(N,N);
s = size(data1,1);
hx = (xmax-xmin)/N;
hy = (ymax-ymin)/N;
for i=1:s
    c1 = round((data1(i) - xmin) / hx);
    c2 = round((data2(i) - ymin) / hy);
    bins(c1,c2) = bins(c1,c2) + ws(i);
end
x = xmin+hx/2:hx:xmax-hx/2;
y = ymin+hy/2:hy:ymax-hy/2;