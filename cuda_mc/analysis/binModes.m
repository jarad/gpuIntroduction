function counts = binModes(data,h)
N = size(data,1);
mus = perms([-3 0 3 6]);
P = size(mus,1);
counts = zeros(P,1);
for i=1:N
    for j=1:P
        if norm(data(i,:) - mus(j,:)) < h
            counts(j) = counts(j) + 1;
        end
    end
end