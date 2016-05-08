#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cudaErrorHandling.h"

texture<float, 2, cudaReadModeElementType> tex_input;

__global__ void filter_transform_2x2(int C, int K, int stride, float *dev_filter, float *dev_filter_t)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < stride) {
		int idy = id / C;
		int idx = id % C;
		int base = id * 3;
		int outBase = idx*K+idy;

		float a00, a01, a02, a03,
		      a10, a11, a12, a13,
		      a20, a21, a22, a23,
		      a30, a31, a32, a33;

		float3 r0, r1, r2;
		
		r0 = ((float3*)dev_filter)[base];
		r1 = ((float3*)dev_filter)[base + 1];
		r2 = ((float3*)dev_filter)[base + 2];

		float t0 = r0.x + r0.z;
		a00 = r0.x;
		a01 = (t0+r0.y)*0.5;
		a02 = (t0-r0.y)*0.5;
		a03 = r0.z;
		
		float t1 = (r0.y+r1.y+r2.y)*0.5;
		a10 = (r0.x+r1.x+r2.x)*0.5;
		a13 = (r0.z+r1.z+r2.z)*0.5;
		t0 = a10 + a13;
		a11 = (t0+t1)*0.5;
		a12 = (t0-t1)*0.5;
		
		t1 = (r0.y-r1.y+r2.y)*0.5;
		a20 = (r0.x-r1.x+r2.x)*0.5;
		a23 = (r0.z-r1.z+r2.z)*0.5;
		t0 = a20 + a23;
		a21 = (t0+t1)*0.5;
		a22 = (t0-t1)*0.5;

		t0 = r2.x+r2.z;
		a30 = r2.x;
		a31 = (t0+r2.y)*0.5;
		a32 = (t0-r2.y)*0.5;
		a33 = r2.z;

		dev_filter_t[outBase            ] = a00;
		dev_filter_t[outBase +    stride] = a01;
		dev_filter_t[outBase +  2*stride] = a02;
		dev_filter_t[outBase +  3*stride] = a03;
		dev_filter_t[outBase +  4*stride] = a10;
		dev_filter_t[outBase +  5*stride] = a11;
		dev_filter_t[outBase +  6*stride] = a12;
		dev_filter_t[outBase +  7*stride] = a13;
		dev_filter_t[outBase +  8*stride] = a20;
		dev_filter_t[outBase +  9*stride] = a21;
		dev_filter_t[outBase + 10*stride] = a22;
		dev_filter_t[outBase + 11*stride] = a23;
		dev_filter_t[outBase + 12*stride] = a30;
		dev_filter_t[outBase + 13*stride] = a31;
		dev_filter_t[outBase + 14*stride] = a32;
		dev_filter_t[outBase + 15*stride] = a33;
	
		//printf("id %d  %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n", id, a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33);
	}
}
__global__ void data_transform_2x2(int C, int H, int W, int blockSize, float* dev_data)
{
	int idx = threadIdx.x*2;
	int idy = blockIdx.y*H + threadIdx.y*2;
	int outBlockSize = blockSize * C;
	int outBase = (threadIdx.y*blockDim.x+threadIdx.x) + blockIdx.y * blockSize;

	float a00, a01, a02, a03,
	      a10, a11, a12, a13,
	      a20, a21, a22, a23,
	      a30, a31, a32, a33,
	      x10, x13,
	      t00, t10, t20, t30,
	      t01, t11, t21, t31;

	a00 = tex2D(tex_input, idx,   idy);
	a01 = tex2D(tex_input, idx+1, idy);
	a02 = tex2D(tex_input, idx+2, idy);
	a03 = tex2D(tex_input, idx+3, idy);
	a10 = tex2D(tex_input, idx,   idy+1);
	a11 = tex2D(tex_input, idx+1, idy+1);
	a12 = tex2D(tex_input, idx+2, idy+1);
	a13 = tex2D(tex_input, idx+3, idy+1);
	a20 = tex2D(tex_input, idx,   idy+2);
	a21 = tex2D(tex_input, idx+1, idy+2);
	a22 = tex2D(tex_input, idx+2, idy+2);
	a23 = tex2D(tex_input, idx+3, idy+2);
	a30 = tex2D(tex_input, idx,   idy+3);
	a31 = tex2D(tex_input, idx+1, idy+3);
	a32 = tex2D(tex_input, idx+2, idy+3);
	a33 = tex2D(tex_input, idx+3, idy+3);

/*
	if (idy == 0 && idx == 0) {
		printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n", a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33);
	}
*/

	x10 = a10;
	x13 = a13;
	
	t00 =  a01 - a21;
	t10 =  a11 + a21;
	t20 = -a11 + a21;
	t30 =  a11 - a31;

	t01 =  a02 - a22;
	t11 =  a12 + a22;
	t21 = -a12 + a22;
	t31 =  a12 - a32;

/********************************************/

	a00 =  a00 - a20 - t01;
	a01 =  t00 + t01;
	a02 = -t00 + t01;
	a03 =  t00 - a03 + a23;
	
	a10 =  a10 + a20 - t11;
	a11 =  t10 + t11;
	a12 = -t10 + t11;
	a13 =  t10 - a13 - a23;

	a20 = -x10 + a20 - t21;
	a21 =  t20 + t21;
	a22 = -t20 + t21;
	a23 =  t20 + x13 - a23;

	a30 =  x10 - a30 - t31;
	a31 =  t30 + t31;
	a32 = -t30 + t31;
	a33 =  t30 - x13 + a33;

/*
	if (idy == 0 && idx == 0) {
		printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n", a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33);
	}

	printf("outBase %d\n", outBase);
*/

	dev_data[outBase                  ] = a00;
	dev_data[outBase +    outBlockSize] = a01;
	dev_data[outBase +  2*outBlockSize] = a02;
	dev_data[outBase +  3*outBlockSize] = a03;
	dev_data[outBase +  4*outBlockSize] = a10;
	dev_data[outBase +  5*outBlockSize] = a11;
	dev_data[outBase +  6*outBlockSize] = a12;
	dev_data[outBase +  7*outBlockSize] = a13;
	dev_data[outBase +  8*outBlockSize] = a20;
	dev_data[outBase +  9*outBlockSize] = a21;
	dev_data[outBase + 10*outBlockSize] = a22;
	dev_data[outBase + 11*outBlockSize] = a23;
	dev_data[outBase + 12*outBlockSize] = a30;
	dev_data[outBase + 13*outBlockSize] = a31;
	dev_data[outBase + 14*outBlockSize] = a32;
	dev_data[outBase + 15*outBlockSize] = a33;

}

__global__ void inverse_transform_2x2(int N, int K, int stride, int outW, float *d_UV, float *d_output)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < stride) {
		int n = id % N;
		int k = id / N;
		int s0 = k*N*2 + (n/outW)*2*outW + (n%outW);
		int s1 = s0 + outW;

		float a00, a01, a02, a03,
		      a10, a11, a12, a13,
		      a20, a21, a22, a23,
		      a30, a31, a32, a33,
		      t00, t10,
		      t01, t11;
		float2 r0, r1;

		a00 = d_UV[id];
		a01 = d_UV[id +    stride];
		a02 = d_UV[id +  2*stride];
		a03 = d_UV[id +  3*stride];
		a10 = d_UV[id +  4*stride];
		a11 = d_UV[id +  5*stride];
		a12 = d_UV[id +  6*stride];
		a13 = d_UV[id +  7*stride];
		a20 = d_UV[id +  8*stride];
		a21 = d_UV[id +  9*stride];
		a22 = d_UV[id + 10*stride];
		a23 = d_UV[id + 11*stride];
		a30 = d_UV[id + 12*stride];
		a31 = d_UV[id + 13*stride];
		a32 = d_UV[id + 14*stride];
		a33 = d_UV[id + 15*stride];

		t00 = a01 + a11 + a21;
		t01 = a02 + a12 + a22;
		t10 = a11 - a21 - a31;
		t11 = a12 - a22 - a32;

		a00 = a00 + a10 + a20 + t00 + t01;
		a01 = t00 - t01 - a03 - a13 - a23;
		a10 = a10 - a20 - a30 + t10 + t11;
		a11 = t10 - t11 - a13 + a23 + a33;

		r0 = make_float2(a00, a01);
		r1 = make_float2(a10, a11);
		((float2*)d_output)[s0] = r0;
		((float2*)d_output)[s1] = r1;
	}
}

void winograd(int C, int K, int H, int W, const float* h_input, const float* h_filter, float* h_output)
{
	int blockSize = ((H-2)/2)*((W-2)/2);
	dim3 dimBlock((W-2)/2, (H-2)/2, 1); // assume that H-2 and W-2 are both multiple of 2
	dim3 dimGrid(1, C, 1);
	
	/*********************** tex_input ******************************/
	cudaChannelFormatDesc inputDesc = cudaCreateChannelDesc<float>();
	cudaArray* d_inputArray;
	err_handling( cudaMallocArray(&d_inputArray, &inputDesc, W, H*C) );
	err_handling( cudaMemcpyToArray(d_inputArray, 0, 0, h_input, C*H*W*sizeof(float), cudaMemcpyHostToDevice) );
	err_handling( cudaBindTextureToArray(tex_input, d_inputArray) );

	/*********************** dev_output *****************************/
	float *d_output = NULL;
	err_handling( cudaMalloc(&d_output, blockSize*K*4*sizeof(float)) );
	
	/*********************** dev_data *******************************/
	float* h_data = (float*)malloc(C*blockSize*16*sizeof(float));
	float* d_data = NULL;
	err_handling( cudaMalloc(&d_data, C*blockSize*16*sizeof(float)) );

	/*********************** dev_filter *****************************/

	float* d_filter = NULL;
	err_handling( cudaMalloc(&d_filter, C*K*9*sizeof(float)) );
	err_handling( cudaMemcpy(d_filter, h_filter, C*K*9*sizeof(float), cudaMemcpyHostToDevice) );

	float* h_filter_t = (float*)malloc(C*K*16*sizeof(float));
	float* d_filter_t = NULL;
	err_handling( cudaMalloc(&d_filter_t, C*K*16*sizeof(float)) );

	/*********************** dev_UV **********************************/
	float *h_UV = (float*)malloc(blockSize*K*16*sizeof(float));
	float* d_UV = NULL;
	err_handling( cudaMalloc(&d_UV, blockSize*K*16*sizeof(float)) );

	/*********************** cublas *********************************/

	cublasHandle_t handle;
	cublasCreate(&handle);
	float alpha = 1, beta = 0;

	float **d_A, **d_B, **d_C;
	err_handling( cudaMalloc(&d_A, 16*sizeof(float*)) );
	err_handling( cudaMalloc(&d_B, 16*sizeof(float*)) );
	err_handling( cudaMalloc(&d_C, 16*sizeof(float*)) );


	float *h_A[16], *h_B[16], *h_C[16];
	for (int i = 0; i < 16; i++) {
		h_A[i] = &d_data[i*C*blockSize];
		h_B[i] = &d_filter_t[i*C*K];
		h_C[i] = &d_UV[i*blockSize*K];
	}
	
	err_handling( cudaMemcpy(d_A, h_A, 16*sizeof(float*), cudaMemcpyHostToDevice) );
	err_handling( cudaMemcpy(d_B, h_B, 16*sizeof(float*), cudaMemcpyHostToDevice) );
	err_handling( cudaMemcpy(d_C, h_C, 16*sizeof(float*), cudaMemcpyHostToDevice) );

/***************************************************************************************************/
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	filter_transform_2x2<<<(K*C-1)/256+1, 256>>>(C, K, C*K, d_filter, d_filter_t);


	data_transform_2x2<<<dimGrid, dimBlock>>>(C, H, W, blockSize, d_data);


	//pulldown(d_data, C*blockSize*16, "2-data.out");
	//pulldown(d_filter_t, C*K*9, "2-filter_t.out");


	cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, blockSize, K, C,
			   &alpha, (const float **)d_A, blockSize, (const float **)d_B, K, 
			   &beta, d_C, blockSize, 16);

	//pulldown(d_UV, blockSize*K*16, "cublas-UV.out");


	inverse_transform_2x2<<<(blockSize*K-1)/256+1, 256>>>(blockSize, K, blockSize*K, (W-2)/2, d_UV, d_output);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time_elapsed;
	cudaEventElapsedTime(&time_elapsed, start, stop);
	printf("time %fms\n", time_elapsed);

/****************************************************************************************************/

	err_handling( cudaMemcpy(h_output, d_output, blockSize*K*4*sizeof(float), cudaMemcpyDeviceToHost) );

	FILE *fout = fopen("cublas.out", "w");
	for (int i = 0; i < blockSize*K*4; i++) {
		fprintf(fout, "%f\n", h_output[i]);
	}
	fclose(fout);


/****************************************************************************************************/

	err_handling( cudaUnbindTexture(tex_input)  );
	err_handling( cudaFreeArray(d_inputArray) );
	err_handling( cudaFree(d_data) );
	err_handling( cudaFree(d_output) );
	free(h_data);
	free(h_UV);
	cublasDestroy(handle);
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
	int outputSize = K*H*W;
	int filterSize = K*C*9;

	float *data   = (float*)malloc(dataSize*sizeof(float));
	float *output = (float*)malloc(outputSize*sizeof(float));
	float *filter = (float*)malloc(filterSize*sizeof(float));

	if (data == NULL || output == NULL || filter == NULL)
		printf("allocate host err!\n");


	for (int i = 0; i < dataSize; i++) {
		data[i]   = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}
	for (int i = 0; i < filterSize; i++) {
		filter[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}
	for (int i = 0; i < outputSize; i++) {
		output[i] = rand()/(float)RAND_MAX - rand()/(float)RAND_MAX;
	}

/*
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			printf("%.2f ", input[i*W+j]);
		}
		printf("\n");
	}
*/

/***************************************************/
	winograd(C, K, H, W, data, filter, output);
/***************************************************/

	err_handling( cudaDeviceReset() );

	return 0;
}

