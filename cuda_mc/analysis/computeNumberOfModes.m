function counts = computeNumberOfModes(data)
N = size(data,1);
mus = perms([-3 0 3 6]);
P = size(mus,1);
counts = zeros(P,1);
h = 10;
for i=1:N
    for j=1:P
        counts(j) = counts(j) + K(data(i,:),mus(j,:),h);
    end
end