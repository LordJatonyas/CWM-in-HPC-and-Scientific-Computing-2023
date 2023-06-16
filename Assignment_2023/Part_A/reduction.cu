#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda.h>
#include <curand.h>

#define NUM_ELS 1024

__global__ void reduction(float *d_input, float *d_output)
{
    // Allocate shared memory

    __shared__  float smem_array[NUM_ELS];

    // First add during Load to prevent Idle threads
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x* 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    // first, each thread loads data into shared memory

    smem_array[tid] = d_input[tid];

    // next, we perform binary tree reduction
    // to improve performance, I have opted to do loop unrolling
    // and run multiple elements per thread
    // (sequential addressing to resolve bank conflict)

    while (i < NUM_ELS) {
        smem_array[tid] += d_input[i] + d_input[i + blockDim.x];
	i += gridSize;
    }
    __syncthreads();

    if (blockDim.x >= 512) {
	if (tid < 256) smem_array[tid] += smem_array[tid + 256];
	__syncthreads();
    }
    if (blockDim.x >= 256) {
	if (tid < 128) smem_array[tid] += smem_array[tid + 128];
	__syncthreads();
    }
    if (blockDim.x >= 128) {
	if (tid < 64) smem_array[tid] += smem_array[tid + 64];
	__syncthreads();
    }

    // Warp Reduction
    if (tid < 32) {
	if (blockDim.x >= 64) smem_array[tid] += smem_array[tid + 32];
	if (blockDim.x >= 32) smem_array[tid] += smem_array[tid + 16];
	if (blockDim.x >= 16) smem_array[tid] += smem_array[tid + 8];
        if (blockDim.x >= 8) smem_array[tid] += smem_array[tid + 4];
	if (blockDim.x >= 4) smem_array[tid] += smem_array[tid + 2];
	if (blockDim.x >= 2) smem_array[tid] += smem_array[tid + 1];
    }
	
    // finally, first thread puts result into global memory

    if (tid==0) d_output[blockIdx.x] = smem_array[0];
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
    int num_els, num_threads, mem_size;

    float *h_data;
    float *d_input, *d_output;

    // initialise card

    num_els     = NUM_ELS;
    num_threads = (num_els > 1024) ? 1024 : num_els;
    mem_size    = sizeof(float) * num_els;

    // allocate host memory to store the input data
    // and initialize to integer values between 0 and 1000

    h_data = (float*) malloc(mem_size);
      
    for(int i = 0; i < num_els; i++) {
        h_data[i] = ((float)rand()/(float)RAND_MAX);
    }

    // allocate device memory input and output arrays

    cudaMalloc((void**)&d_input, mem_size);
    cudaMalloc((void**)&d_output, sizeof(float));

    // copy host memory to device input array

    cudaMemcpy(d_input, h_data, mem_size, cudaMemcpyHostToDevice);

    // execute the kernel

    // we need an integer for the number of blocks needed to execute reduction
    reduction<<<(int) num_els/num_threads,num_threads>>>(d_input,d_output);

    // copy result from device to host

    cudaMemcpy(h_data, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // check results

    printf("reduction error = %f\n",h_data[0]/NUM_ELS);

    // cleanup memory

    free(h_data);
    cudaFree(d_input);
    cudaFree(d_output);

    // CUDA exit -- needed to flush printf write buffer

    cudaDeviceReset();
}

