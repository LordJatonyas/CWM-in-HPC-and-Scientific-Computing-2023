// In this assignment you will expand your "Hello world" kernel to see how
// are threads, warps and blocks scheduled.
//
// Follow instructions for TASK 1 which consists from writing a kernel,
// configuring it and then running the code. After running the code few 
// times you should see that blocks are executed in no particular order
//
// After you finish TASK 1 continue with TASK 2 and TASK 3 following same
// workflow. Write the kernel, configure it properly and then run code 
// multiple times to see how threads from one warp are schedules and how
// warps from one block are scheduled.

// NOTE: You should finish your basic "Hello world" assignment first, before 
//       doing this one.

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


//----------------------------------------------------------------------
// TASK 1.0: Write a new "Hello world" kernel, called for example 
// 'helloworld_blocks', which in addition to "Hello world" writes out which
// block is writing out the string. 
// For example "Hello world from block 2!"
// 
// In order to print which block is saying "Hello world" you can use syntax
// like this: 
// printf("integer=%d; float=%f or %e;\n",1, 0.0001, 0.0001);
// Also remember that every thread can access pre-set variable which
// refer to its coordinates and coordinates of the block which it resides in.
// These are dim3 data types called: threadIdx, blockIdx, blockDim
//                                   and gridDim
// dim3 data type has three components: x, y, z

// write your kernel here

//----------------------------------------------------------------------
__global__ void helloworld_blocks_kernel(float *) {
  printf("Hello world from block %d\n", blockIdx.x);
}


//----------------------------------------------------------------------
// TASK 2.0: Write a "Hello world" kernel which output "Hello world" but 
//          in addition to that also outputs which block and thread it
//          comes from. For example: "Hello world from block 1, thread 3"
// 
// As in task one use printf() function to print to console and utilise
// pre-set variables threadIdx, blockIdx, blockDim and gridDim.

// write your kernel here

//----------------------------------------------------------------------
__global__ void helloworld_blocks_and_threads_kernel(float *) {
  printf("Hello world from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}


//----------------------------------------------------------------------
// TASK 3.0: Write a "Hello world" kernel where only first thread from each
//           warp writes out to console. So for example:
//           "Hello world from block 2, warp 3"
// 
// A warp is group of 32 threads. First warp is consists from threads 0--31, 
// second warp consists from threads 32--63 and so on. To select first thread
// from each warp we have to use modulo "%" operation. Modulo operation returns
// remainder after division. So 3%2=1 while 4&2=0;
// To select first thread from each warp we need to use a branch like this:
// if(threadIdx.x%32==0) {
//   this block will be executed only by first thread from each warp
// }
// To identify which warp thread resides in you should remember that warp consist
// from 32 threads.

// write your kernel here

//----------------------------------------------------------------------
__global__ void helloworld_blocks_and_warps_kernel(float *) {
  int warp = 1 + (int) (threadIdx.x / 32);
  if (!(threadIdx.x % 32)) {
    printf("Hello world from block %d, warp %d\n", blockIdx.x, warp);
  }
}

int main(void) {
  // initiate GPU
  int deviceid = 0;
  int devCount;
  cudaGetDeviceCount(&devCount);
  if(deviceid<devCount){
    cudaSetDevice(deviceid);
  }
  else {
    printf("ERROR! Selected device is not available\n");
    return(1);
  }
	
  //----------------------------------------------------------------------
  // TASK 1.1: execute your "Hello world" kernel from TASK 1.0 on few blocks 
  // (10 should be enough) with 1 thread. When you had configured your
  // kernel compile the code typing "make" and then run it be executing
  // ./helloworld_scheduling.exe
  // You should see that blocks are scheduled in haphazard manner.
  // 
  // You may use whatever syntax version you prefer, a simplified one 
  // dimensional or full three dimensional call using dim3 data type.
  
  // put your code here
  
  //----------------------------------------------------------------------
  /*
  float *h_x, *d_x;
  int nblocks = 10, nthreads = 1, nsize = 10 * 1;
  h_x = (float *) malloc(nsize * sizeof(float));
  cudaMalloc((void **) &d_x, nsize * sizeof(float));
  helloworld_blocks_kernel<<<nblocks, nthreads>>>(d_x);
  cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  free(h_x);
  */

  //----------------------------------------------------------------------
  // TASK 2.1: execute your "Hello world" kernel from TASK 2.0 on about  
  // 5 blocks each containing about 10 threads. When you configured the kernel
  // compile the code typing "make" and then run it be executing
  // ./helloworld_scheduling.exe
  // You should see that blocks are still scheduled in haphazard manner,
  // but threads within them, being from one warp should execute in order. 
  // 
  // You may use whatever syntax version you prefer, a simplified one 
  // dimensional or full three dimensional call using dim3 data type.
  
  // put your code here
  
  //----------------------------------------------------------------------
  /*
  float *h_x, *d_x;
  int nblocks = 5, nthreads = 10, nsize = 5 * 10;
  h_x = (float *) malloc(nsize * sizeof(float));
  cudaMalloc((void **) &d_x, nsize * sizeof(float));
  helloworld_blocks_and_threads_kernel<<<nblocks, nthreads>>>(d_x);
  cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  free(h_x);
  */
  
  //----------------------------------------------------------------------
  // TASK 3.1: execute your "Hello world" kernel from TASK 3.0 on about  
  // 5 blocks each containing about 320 threads. When you configured the kernel
  // compile the code typing "make" and then run it be executing
  // ./helloworld_scheduling.exe
  // You should see that both blocks and warps within them are scheduled
  // in haphazard manner.
  // To see more clearly that warps are executed in haphazard manner run
  // your kernel with only one block.
  // 
  // You may use whatever syntax version you prefer, a simplified one 
  // dimensional or full three dimensional call using dim3 data type.
  
  // put your code here
  
  //----------------------------------------------------------------------
  float *h_x, *d_x;
  int nblocks = 1, nthreads = 320, nsize = 1 * 320;
  h_x = (float *) malloc(nsize * sizeof(float));
  cudaMalloc((void **) &d_x, nsize * sizeof(float));
  helloworld_blocks_and_warps_kernel<<<nblocks, nthreads>>>(d_x);
  cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  free(h_x);
  

  cudaDeviceReset();
  return (0);
}
