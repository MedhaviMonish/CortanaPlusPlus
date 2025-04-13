
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

template <typename T>
void launchAddKernel(T *c, const T *a, const T *b, int numThreads)
{
    addKernel<T><<<1, 500>>>(c, a, b);
}