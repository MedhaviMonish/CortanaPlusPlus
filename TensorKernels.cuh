
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
template <typename T>
__global__ void addKernel(T *a, const T *b, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = a[i] + b[i];
    }
}

template <typename T>
void launchAddKernel(T *a, const T *b, dim3 thread_blocks, dim3 thread_per_blocks, int total_size)
{
    addKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, total_size);
}

template <typename T>
__global__ void matMulKernel(const T *a, const T *b, T *c, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = a[i] + b[i];
    }
}

template <typename T>
void launchMatMulKernel(const T *a, const T *b, T *c, dim3 thread_blocks, dim3 thread_per_blocks,
                        int total_size)
{
    addKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, c, total_size);
}