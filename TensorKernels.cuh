
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
void launchAddKernel(T *c, const T *a, const T *b, dim3 thread_blocks, dim3 thread_per_blocks,
                     int total_size)
{
    addKernel<T><<<thread_blocks, thread_per_blocks>>>(c, a, b, total_size);
}