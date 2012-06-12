function y = iteration_modes(data,h)
N = size(data,1);
mus = perms([-3 0 3 6]);
P = size(mus,1);
y = zeros(N, 1);
for i=1:N
    for j=1:P
        if norm(data(i,:) - mus(j,:)) < h
            y(i,1) = j;
        end
    end
end