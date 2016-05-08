#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float G[4][3] = { {1, 0, 0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0, 0, 1} };
float GT[3][4] = { {1, 0.5, 0.5, 0}, {0, 0.5, -0.5, 0}, {0, 0.5, 0.5, 1} };

float BT[4][4] = { {1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
float B[4][4] = { {1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};

float AT[2][4] = { {1, 1, 1, 0}, {0, 1, -1, -1} };
float A[4][2] = { {1, 0}, {1, 1}, {1, -1}, {0, -1} };

void filter_transform_2x2(int K, int C, float* input, float* output)
{
	float b[4][3];
	float y[4][4];
	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			float *filter = input + (k*C+c)*3*3;
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 3; j++) {
					float val = 0;
					for (int t = 0; t < 3; t++) {
						val += G[i][t] * filter[t*3+j];
					}
					b[i][j] = val;
				}
			}

			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					float val = 0;
					for (int t = 0; t < 3; t++) {
						val += b[i][t] * GT[t][j];
					}
					y[i][j] = val;
				}
			}
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					output[c*K+k + (i*4+j)*K*C] = y[i][j];
				}
			}
		}
	}
}

void data_transform_2x2(int C, int H, int W, const float* input, float* output)
{
	float a[4][4];
	float b[4][4];
	float y[4][4];
	int outH = (H-2)/2;
	int outW = (W-2)/2;
	for (int c = 0; c < C; c++) {
		for (int i = 0; i < outH; i++) {
			for (int j = 0; j < outW; j++) { // iter output index
				for (int ii = 0; ii < 4; ii++) {
					for (int jj = 0; jj < 4; jj++) { // read sub tile
						a[ii][jj] = input[c*H*W + (i*2+ii)*W + j*2+jj];
					}
				}
				

				for (int ii = 0; ii < 4; ii++) { // matrix multiply
					for (int jj = 0; jj < 4; jj++) {
						float sum = 0;
						for (int k = 0; k < 4; k++) {
							sum += BT[ii][k] * a[k][jj];
						}
						b[ii][jj] = sum;
					}
				}

				for (int ii = 0; ii < 4; ii++) { // matrix multiply
					for (int jj = 0; jj < 4; jj++) {
						float sum = 0;
						for (int k = 0; k < 4; k++) {
							sum += b[ii][k] * B[k][jj];
						}
						y[ii][jj] = sum;
					}
				}

				for (int ii = 0; ii < 4; ii++) {
					for (int jj = 0; jj < 4; jj++) {
						output[c*(outH*outW) + (i*outW+j) + (ii*4+jj)*(outH*outW)*C] = y[ii][jj];
					}
				}
				
			}
		}
	}
}

void matmul(int M, int N, int K, const float *A, const float *B, float *C)
{
	printf("M,N,K %d %d %d\n", M, N, K);
	int strideA = M*K;
	int strideB = K*N;
	int strideC = M*N;
	for (int t = 0; t < 16; t++) {
		
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < N; j++) {
				float val = 0;
				for (int k = 0; k < K; k++) {
					val += A[k*M+i] * B[k*N+j];
				}
				C[j*M+i] = val;
			}
		}

		A += strideA;
		B += strideB;
		C += strideC;
	}
}

void inverse_transform_2x2(int K, int H, int W, const float* input, float* output)
{
	float a[4][4];
	float b[2][4];
	float y[2][2];
	int outH = (H-2)/2;
	int outW = (W-2)/2;
	int N = outH * outW;
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < N; j++) {
			for (int t = 0; t < 16; t++) {
				a[t/4][t%4] =  input[t*K*N + i*N+j];
			}
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 4; jj++) {
					float val = 0;
					for (int k = 0; k < 4; k++) {
						val += AT[ii][k] * a[k][jj];
					}
					b[ii][jj] = val;
				}
			}

			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					float val = 0;
					for (int k = 0; k < 4; k++) {
						val += b[ii][k] * A[k][jj];
					}
					y[ii][jj] = val;
				}
			}

			int outh = j / outW;
			int outw = j % outW;
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					output[i*outH*outW*4 + (outh*2+ii)*outW*2 + outw*2+jj] = y[ii][jj];
				}
			}
		}
	}
}


void winograd(int C, int K, int H, int W, float* data, float* filter, float* output)
{
	int blockSize = ((H-2)/2)*((W-2)/2);

	float* data_t   = (float*)malloc(C*blockSize*16*sizeof(float));
	float* filter_t = (float*)malloc(K*C*16*sizeof(float));
	float* UV       = (float*)malloc(blockSize*K*16*sizeof(float));
	memset(UV, 0, blockSize*16*sizeof(float));

	
	filter_transform_2x2(K, C, filter, filter_t);
	printf("filter\n");
	

	data_transform_2x2(C, H, W, data, data_t);
	printf("data\n");


	matmul(blockSize, K, C, data_t, filter_t, UV);
	printf("mat\n");


	inverse_transform_2x2(K, H, W, UV, output);
	printf("inverse\n");


	FILE *fp = fopen("2-1.out", "w");
	for (int i = 0; i < blockSize*K*4; i++) {
		fprintf(fp, "%f\n", output[i]);
	}
	fclose(fp);

	/*********************** write data to file *********************/

	FILE *fout = fopen("2.out", "w");
	for (int i = 0; i < blockSize*K*4; i++) {
		fprintf(fout, "%f\n", output[i]);
	}
	fclose(fout);

	free(UV);
	free(data_t);
	free(filter_t);
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
	float *filter = (float*)malloc(filterSize*sizeof(float));
	float *output = (float*)malloc(outputSize*sizeof(float));

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


/*
	data[0] = 1;
	data[1] = 2;
	data[2] = 3;
	data[3] = 4;

	data[16] = 5;
	data[17] = 6;
	data[18] = 7;
	data[19] = 8;

	data[32] = 9;
	data[33] = 10;
	data[34] = 11;
	data[35] = 12;

	data[48] = 13;
	data[49] = 14;
	data[50] = 15;
	data[51] = 16;
*/

/*
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			printf("%.2f ", data[i*W+j]);
		}
		printf("\n");
	}
*/
/***************************************************/
	winograd(C, K, H, W, data, filter, output);
/***************************************************/

	
	return 0;
}

