function w = norm_weights(ws)
N = size(ws,1);
s = sum(ws);
w = ws * N / s;
