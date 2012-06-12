function frames = analyze_smcs(N, T, D, data, xmin, xmax, ymax, h)
%M = N * T;
xs = zeros(T,N,D);
frames = moviein(T);



for i=1:T
    j1 = N*(i-1) + 1;
    j2 = i*N;
    xs(i,:,:) = data(j1:j2,:);
    hist(xs(i,:,1),xmin:h:xmax);
    axis([xmin xmax 0 ymax]);
    frames(i,:) = getframe;
end

%movie(frames,1,10);

