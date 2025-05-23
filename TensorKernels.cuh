
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
__global__ void subKernel(T *a, const T *b, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = a[i] - b[i];
    }
}

template <typename T>
void launchSubKernel(T *a, const T *b, dim3 thread_blocks, dim3 thread_per_blocks, int total_size)
{
    subKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, total_size);
}

template <typename T>
__global__ void addScalarKernel(T *a, T value, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = a[i] + value;
    }
}

template <typename T>
void launchAddScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks, T value, int N)
{
    addScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T>
__global__ void subScalarKernel(T *a, T value, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = a[i] - value;
    }
}

template <typename T>
void launchSubScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks, T value, int N)
{
    subScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T>
__global__ void multiplyKernel(T *a, const T *b, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = a[i] * b[i];
    }
}

template <typename T>
void launchMultiplyKernel(T *a, const T *b, dim3 thread_blocks, dim3 thread_per_blocks,
                          int total_size)
{
    multiplyKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, total_size);
}

template <typename T>
__global__ void multiplyScalarKernel(T *a, T value, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = a[i] * value;
    }
}

template <typename T>
void launchMultiplyScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks, T value, int N)
{
    multiplyScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T>
__global__ void divKernel(T *a, const T *b, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = a[i] / b[i];
    }
}

template <typename T>
void launchDivKernel(T *a, const T *b, dim3 thread_blocks, dim3 thread_per_blocks, int total_size)
{
    divKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, total_size);
}

template <typename T>
__global__ void divScalarKernel(T *a, T value, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = a[i] / value;
    }
}

template <typename T>
void launchDivScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks, T value, int N)
{
    divScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T>
__global__ void powScalarKernel(T *a, T value, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = pow(a[i], value); // uses device-side pow()
    }
}

template <typename T>
void launchPowScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks, T value, int N)
{
    powScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}

template <typename T>
__global__ void matMulKernel(const T *a, const T *b, T *c)
{
    // Custom CUDA kernel for broadcasted elementwise matmul without loops
    // Launches 1 block per (weight_row, feature) and 1 thread per input row
    // Computes C[i][j][f] = A[i][f] * B[j][f] with flat memory layout
    // i: input row index (threadIdx.x) � selects each input sample
    // j: weight row index (blockIdx.x) � selects each row of the weight matrix
    // k: feature column index (blockIdx.y) � selects each feature column
    // a[i]: refers to A[i][k] � value from input row at feature k
    // b[j]: refers to B[j][k] � value from weight row at feature k
    int i = threadIdx.x * gridDim.y + blockIdx.y;    // all rows of input A
    int j = blockIdx.x * gridDim.y + blockIdx.y;     // all elements of B
    int k = threadIdx.x * gridDim.x * gridDim.y + j; // for each input row
    c[k] = a[i] * b[j];
}

template <typename T>
void launchMatMulKernel(const T *a, const T *b, T *c, dim3 thread_blocks, dim3 thread_per_blocks)
{
    matMulKernel<T><<<thread_blocks, thread_per_blocks>>>(a, b, c);
}

template <typename T>
__global__ void reduceSumLastAxisKernel(T *a, T *s, int stride, int lastRowSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int limit;
    if (threadIdx.x == blockDim.x - 1)
    {
        limit = lastRowSize - threadIdx.x * stride;
    }
    else
    {
        limit = stride;
    }
    int i = blockIdx.x * lastRowSize + threadIdx.x * stride;
    int end = i + limit;
    s[j] = 0;
    for (; i < end; ++i)
    {
        s[j] += a[i];
    }
}

template <typename T>
void launchReduceSumLastAxisKernel(T *a, T *s, dim3 thread_blocks, dim3 thread_per_blocks,
                                   int stride, int lastRowSize)
{
    reduceSumLastAxisKernel<T><<<thread_blocks, thread_per_blocks>>>(a, s, stride, lastRowSize);
}

template <typename T>
__global__ void maxScalarKernel(T *a, T value, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
    {
        a[i] = (a[i] > value) ? a[i] : value;
    }
}

template <typename T>
void launchMaxScalarKernel(T *a, dim3 thread_blocks, dim3 thread_per_blocks, T value, int N)
{
    maxScalarKernel<T><<<thread_blocks, thread_per_blocks>>>(a, value, N);
}