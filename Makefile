all: cpu-wino gpu-wino cpu-std gpu-wino-mygemm

cpu-% : cpu-%.c
	gcc -std=c99 -O3 -o $@ $<

gpu-% : gpu-%.cu cudaErrorHandling.h
	nvcc -O3 --ptxas-options=-v -arch=sm_50 -lcublas -o $@ $<
