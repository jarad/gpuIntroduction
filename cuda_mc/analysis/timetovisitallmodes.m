function t = timetovisitallmodes(im, M)
N = size(im, 1);
v = zeros(M, 1);
j = 1;
t = 0;
for i = 1:N
    if im(i) ~= 0
        v(im(i)) = 1;
    end
    if (v == ones(M,1))
        t(j,1) = i;
        j = j + 1;
        v = zeros(M,1);
        v(im(i)) = 1;
    end
end
t = cat(1, t(1), diff(t));