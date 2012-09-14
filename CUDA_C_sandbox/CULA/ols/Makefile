all: ols

ols: ols.cu
	nvcc -lcublas ols.cu -o ols

run: ols
	./ols

clean:
	rm *.o 