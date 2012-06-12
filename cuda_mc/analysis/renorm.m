function b = renorm(bins)
M = max(max(bins(:,:)));
N = size(bins,1);
b = zeros(N,N);
for i = 1:N
    for j = 1:N
        b(i,j) = 1 - bins(i,j) / M;
    end
end
