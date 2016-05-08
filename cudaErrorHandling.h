/*
void err_handling(cudaError_t err, const char *str)
{
	if (err != cudaSuccess) {
		printf("%s\n", str);
		exit(EXIT_FAILURE);
	}
}
*/


#define err_handling(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void pulldown(float *dev, int size, char *name)
{
	float *host = (float*)malloc(size*sizeof(float));
	err_handling( cudaMemcpy(host, dev, size*sizeof(float), cudaMemcpyDeviceToHost) );
	FILE *fout = fopen(name, "w");
	for (int i = 0; i < size; i++) {
		fprintf(fout, "%f\n", host[i]);
	}

	fclose(fout);
	free(host);
}
