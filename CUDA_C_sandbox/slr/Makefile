all:
	nvcc -I /usr/local/cula/include -L /usr/local/cula/lib64 -lcula_core -lcula_lapack -lcublas -lcudart slr.cu -o slr

clean:
	rm slr