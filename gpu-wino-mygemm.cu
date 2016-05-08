#include <stdio.h>
#include <cuda_runtime.h>
#include "cudaErrorHandling.h"

texture<float, 3, cudaReadModeElementType> tex_A;
texture<float, 3, cudaReadModeElementType> tex_B;

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

__global__ void matMul(float *C, int M, int K, int N, int pitch)
{
	__shared__ float sA_bf[2][8*64];
	__shared__ float sB_bf[2][8*64];
	float *A_pref, *A_now;
	float *B_pref, *B_now;

	int x = threadIdx.x;
	int y = threadIdx.y;

	int bx = blockIdx.x*64;
	int by = blockIdx.y*64;
	int batch_id = blockIdx.z;
	
	int id = y*8+x;
	int inv_id = ((id&28)<<1) + (id%4) + (id<32? 0:4); //id%32/4*8
	int glbA_id = by + inv_id;
	int glbB_id = bx + inv_id;


	float a0[8], a1[8];
	float b0[8], b1[8];

	float c[8][8];

	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < 8; j++)
			c[i][j] = 0.0;
	
/*********************************************************************/
	for (int i = 0; i < 8; ++i) { // first batch of shared store
		sA_bf[0][i*64+id] = tex3D(tex_A, glbA_id, i, batch_id);
		sB_bf[0][i*64+id] = tex3D(tex_B, glbB_id, i, batch_id);
	}

	A_pref = sA_bf[1];
	B_pref = sB_bf[1];
	A_now  = sA_bf[0];
	B_now  = sB_bf[0];

	int track_bf = 0;


/****************************** main loop ******************************/
	for (int t = 8; t < K; t += 8) {

		__syncthreads();

		A_pref[id] = tex3D(tex_A, glbA_id, t, batch_id); // double buffered shared store
		B_pref[id] = tex3D(tex_B, glbB_id, t, batch_id);

		((float4*)a0)[0] = ((float4*)A_now)[y]; // first shared load of each step
		((float4*)b0)[0] = ((float4*)B_now)[x];
		((float4*)a0)[1] = ((float4*)A_now)[y+8];
		((float4*)b0)[1] = ((float4*)B_now)[x+8];
		
		#pragma unroll
		for (int i = 1; i < 8; ++i) {
			int base = i * 16;
			A_pref[i*64+id] = tex3D(tex_A, glbA_id, t+i, batch_id); // double bufferd shared store
			B_pref[i*64+id] = tex3D(tex_B, glbB_id, t+i, batch_id);

			if (i&1) {
				((float4*)a1)[0] = ((float4*)A_now)[base+y]; // double buffered shared load
				((float4*)b1)[0] = ((float4*)B_now)[base+x];
				((float4*)a1)[1] = ((float4*)A_now)[base+y+8];
				((float4*)b1)[1] = ((float4*)B_now)[base+x+8];

				for (int ii = 0; ii < 8; ++ii)
					for (int jj = 0; jj < 8; ++jj)
						c[ii][jj] += a0[ii] * b0[jj];
				
			} else {
				((float4*)a0)[0] = ((float4*)A_now)[base+y]; // double buffered shared load
				((float4*)b0)[0] = ((float4*)B_now)[base+x];
				((float4*)a0)[1] = ((float4*)A_now)[base+y+8];
				((float4*)b0)[1] = ((float4*)B_now)[base+x+8];

				for (int ii = 0; ii < 8; ++ii)
					for (int jj = 0; jj < 8; ++jj)
						c[ii][jj] += a1[ii] * b1[jj];

			}
		}

		for (int i = 0; i < 8; ++i) { // remained computation of each step
			for (int j = 0; j < 8; ++j) {
				c[i][j] += a1[i] * b1[j];
			}
		}

		A_pref = sA_bf[track_bf]; // shared double buffer pointer exchange
		B_pref = sB_bf[track_bf];
		A_now  = sA_bf[1-track_bf];
		B_now  = sB_bf[1-track_bf];
		track_bf = 1 ^ track_bf; // flip between 0 & 1

	}
	__syncthreads(); // need sync to ensure the last shared store complete

/************************************ remained step *******************************************/

	((float4*)a0)[0] = ((float4*)A_now)[y];
	((float4*)b0)[0] = ((float4*)B_now)[x];
	((float4*)a0)[1] = ((float4*)A_now)[y+8];
	((float4*)b0)[1] = ((float4*)B_now)[x+8];

	#pragma unroll
	for (int i = 1; i < 8; ++i) {
		int base = i * 16;

		if (i&1) {
			((float4*)a1)[0] = ((float4*)A_now)[base+y];
			((float4*)b1)[0] = ((float4*)B_now)[base+x];
			((float4*)a1)[1] = ((float4*)A_now)[base+y+8];
			((float4*)b1)[1] = ((float4*)B_now)[base+x+8];

			for (int ii = 0; ii < 8; ++ii)
				for (int jj = 0; jj < 8; ++jj)
					c[ii][jj] += a0[ii] * b0[jj];

		} else {
			((float4*)a0)[0] = ((float4*)A_now)[base+y];
			((float4*)b0)[0] = ((float4*)B_now)[base+x];
			((float4*)a0)[1] = ((float4*)A_now)[base+y+8];
			((float4*)b0)[1] = ((float4*)B_now)[base+x+8];

			for (int ii = 0; ii < 8; ++ii)
				for (int jj = 0; jj < 8; ++jj)
					c[ii][jj] += a1[ii] * b1[jj];

		}

	}

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			c[i][j] += a1[i] * b1[j];
		}
	}

/********************************** wirte back *****************************************/
	__syncthreads();

/*
	baseSh: base offset for shared memory load
		warp 0 start from  0+id_inwarp
		warp 1 start from 64+id_inwarp

	row:    row number for global write
		warp 0: 0 for first 16 threads; 8 for second 16 threads;
		warp 1: 32 for first 16 threads; 40 for second 16 threads;
*/
	C += batch_id * pitch * M;
	int baseSh = (id<32? 0:64) + (id&31);
	int row = by + ((id&16)>>1) + (id<32? 0:32);

	for (int i = 0; i < 8; ++i) {
		int rowi = row+i;
		((float4*)sA_bf[0])[id*2]   = ((float4*)(c[i]))[0];
		((float4*)sA_bf[0])[id*2+1] = ((float4*)(c[i]))[1];

		if (bx + id%16*4 < pitch) { // bound condition in x direction
			if (rowi < M) // bound condition in y direction
				((float4*)&C[(rowi   )*pitch+bx])[id%16] = ((float4*)sA_bf[0])[baseSh];    // row  0 and  8 | 32 and 40
			if (rowi+16 < M) //bound condition in y direction
				((float4*)&C[(rowi+16)*pitch+bx])[id%16] = ((float4*)sA_bf[0])[baseSh+32]; // row 16 and 24 | 48 and 56
		}
	}
}



__global__ void inverse_transform_2x2(int N, int K, int stride, int outW, float *d_UV, float *d_output)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < stride) {
		//int n = id % N;
		//int k = id / N;
		int k = id % K;
		int n = id / K;
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
void pulldown1(float *dev, char *name, int m, int n)
{
	int size = n*m*16;
	float *host = (float*)malloc(size*sizeof(float));
	err_handling( cudaMemcpy(host, dev, size*sizeof(float), cudaMemcpyDeviceToHost) );
	FILE *fout = fopen(name, "w");

	printf("n %d m %d\n", m, n);
	for (int b = 0; b < 16; b++) {
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m; i++) {
				fprintf(fout, "%f\n", host[b*n*m+i*n+j]);
			}
		}
	}
	fclose(fout);
	free(host);
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
	float* d_data = NULL;
	err_handling( cudaMalloc(&d_data, C*blockSize*16*sizeof(float)) );

	/*********************** dev_filter *****************************/

	float* d_filter = NULL;
	err_handling( cudaMalloc(&d_filter, C*K*9*sizeof(float)) );
	err_handling( cudaMemcpy(d_filter, h_filter, C*K*9*sizeof(float), cudaMemcpyHostToDevice) );

	float* d_filter_t = NULL;
	err_handling( cudaMalloc(&d_filter_t, C*K*16*sizeof(float)) );

	/*********************** dev_UV **********************************/
	float* d_UV = NULL;
	err_handling( cudaMalloc(&d_UV, blockSize*K*16*sizeof(float)) );


	/*********************** prepare for matmul *******************************/

	cudaChannelFormatDesc ADesc = cudaCreateChannelDesc<float>();
	cudaChannelFormatDesc BDesc = cudaCreateChannelDesc<float>();
	cudaArray *A_array, *B_array;
	cudaExtent extentA, extentB;

	extentA = make_cudaExtent(blockSize, C, 16);
	extentB = make_cudaExtent(K, C, 16);

	err_handling(  cudaMalloc3DArray(&A_array, &ADesc, extentA)  );
	err_handling(  cudaMalloc3DArray(&B_array, &BDesc, extentB)  );

	err_handling(  cudaBindTextureToArray(tex_A, A_array)  );
	err_handling(  cudaBindTextureToArray(tex_B, B_array)  );

	tex_A.addressMode[0] = cudaAddressModeBorder;
	tex_A.addressMode[1] = cudaAddressModeBorder;

	tex_B.addressMode[0] = cudaAddressModeBorder;
	tex_B.addressMode[1] = cudaAddressModeBorder;

	dim3 dimGrid_matmul((K-1)/64+1, (blockSize-1)/64+1, 16);
	dim3 dimBlock_matmul(8, 8, 1);

	cudaMemcpy3DParms copyParamsA = {0};
	copyParamsA.srcPtr = make_cudaPitchedPtr((void*)d_data, blockSize*sizeof(float), blockSize, C);
	copyParamsA.dstArray = A_array;
	copyParamsA.extent = extentA;
	copyParamsA.kind = cudaMemcpyDeviceToDevice;

	cudaMemcpy3DParms copyParamsB = {0};
	copyParamsB.srcPtr = make_cudaPitchedPtr((void*)d_filter_t, K*sizeof(float), K, C);
	copyParamsB.dstArray = B_array;
	copyParamsB.extent = extentB;
	copyParamsB.kind = cudaMemcpyDeviceToDevice;

/***************************************************************************************************/
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	filter_transform_2x2<<<(K*C-1)/256+1, 256>>>(C, K, C*K, d_filter, d_filter_t);

	data_transform_2x2<<<dimGrid, dimBlock>>>(C, H, W, blockSize, d_data);


	err_handling(  cudaMemcpy3D(&copyParamsA)  );
	err_handling(  cudaMemcpy3D(&copyParamsB)  );


	matMul<<<dimGrid_matmul, dimBlock_matmul>>>(d_UV, blockSize, C, K, K);


	inverse_transform_2x2<<<(blockSize*K-1)/256+1, 256>>>(blockSize, K, blockSize*K, (W-2)/2, d_UV, d_output);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time_elapsed;
	cudaEventElapsedTime(&time_elapsed, start, stop);
	printf("time %fms\n", time_elapsed);

/****************************************************************************************************/

	err_handling( cudaMemcpy(h_output, d_output, blockSize*K*4*sizeof(float), cudaMemcpyDeviceToHost) );

	FILE *fout = fopen("mygemm.out", "w");
	for (int i = 0; i < blockSize*K*4; i++) {
		fprintf(fout, "%f\n", h_output[i]);
	}
	fclose(fout);


/****************************************************************************************************/

	err_handling(  cudaUnbindTexture(tex_input)  );
	err_handling(  cudaUnbindTexture(tex_A)  );
	err_handling(  cudaUnbindTexture(tex_B)  );
	err_handling(  cudaFreeArray(d_inputArray)  );
	err_handling(  cudaFree(d_data)  );
	err_handling(  cudaFree(d_output)  );
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


/***************************************************/
	winograd(C, K, H, W, data, filter, output);
/***************************************************/

	err_handling( cudaDeviceReset() );

	return 0;
}

