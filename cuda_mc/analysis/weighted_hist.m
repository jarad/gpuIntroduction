function [] = weighted_hist(X,w,a,b,h)
numbins = (b - a)/h;
bins = zeros(numbins,1);
binvals = zeros(numbins,1);

for i = 1:numbins
    binvals(i) = a + i * h;
end
N = size(X,1);
for i=1:N
    j = 1;
    while X(i) > binvals(j)
        j = j + 1;
    end
    bins(j) = bins(j) + w(i);
end

binmeds = zeros(numbins,1);
for i=1:numbins
    binmeds(i) = a + i * h - h / 2;
end
bar(binmeds, bins,1)

    