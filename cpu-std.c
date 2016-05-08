#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void standard(int C, int K, int H, int W, float* data, float* filter, float* output)
{
	for (int k = 0; k < K; k++) {
		for (int h = 0; h < H-2; h++) {
			for (int w = 0; w < W-2; w++) {
				float val = 0;
				for (int c = 0; c < C; c++) {
					for (int ky = 0; ky < 3; ky++) {
						for (int kx = 0; kx < 3; kx++) {
							if (h == 0 && w == 0)printf("%f %f\n", data[(h+ky)*W + (w+kx)], filter[k*C*3*3 + c*3*3 + (ky)*3+(kx)]);
							val += data[c*H*W + (h+ky)*W + (w+kx)] * filter[k*C*3*3 + c*3*3 + (ky)*3+(kx)];
						}
					}
				}
				if (h == 0 && w == 0)printf("val %f\n", val);
				output[k*(H-2)*(W-2) + h*(W-2)+w] = val;
			}
		}
	}

	FILE *fp = fopen("3.out", "w");
	for (int i = 0; i < K*(H-2)*(W-2); i++) {
		fprintf(fp, "%f\n", output[i]);
	}
}

int main(const int argc, const char *argv[])
{
	if (argc != 5) {
		printf("usage: xx.out c k h w\n");
		return 1;
	}
	
	int C = atoi(argv[1]);
	int K = atoi(argv[2]);
	int H = atoi(argv[3]);
	int W = atoi(argv[4]);
	printf("C K H W = %d %d %d %d\n", C, K, H, W);
	
	int dataSize   = C*H*W;
	int outputSize = K*(H-2)*(W-2);
	int filterSize = K*C*3*3;

	float *data   = (float*)malloc(dataSize*sizeof(float));
	float *output = (float*)malloc(outputSize*sizeof(float));
	float *filter = (float*)malloc(filterSize*sizeof(float));

	if (data == NULL || output == NULL || filter == NULL)
		printf("allocate host err!\n");


	for (int i = 0; i < dataSize; i++) {
		data[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}
	for (int i = 0; i < filterSize; i++) {
		filter[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}
	for (int i = 0; i < outputSize; i++) {
		output[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}

	FILE *fp = fopen("3-1.out", "w");
	for (int i = 0; i < dataSize; i++) {
		fprintf(fp, "%f\n", data[i]);
	}
/***************************************************/
	standard(C, K, H, W, data, filter, output);
/***************************************************/

	
	return 0;
}

