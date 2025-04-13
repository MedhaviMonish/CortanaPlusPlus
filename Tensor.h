#pragma once
#include "TensorKernels.cuh"
#include <iostream>
#include <string>
#define MAX_PRINT_THRESHOLD 1000
#define MIN_PRINT_THRESHOLD 6

template <typename T>
class Tensor
{
  private:
    T *data;
    int *shape;
    int dims;
    int total_size;
    Tensor(int *shape, int dims);

  public:
    Tensor(T *data, int *shape, int dims);
    Tensor(const Tensor &other);
    ~Tensor();

    static Tensor<T> getOnes(int *shape, int dims);
    static Tensor<T> getZeroes(int *shape, int dims);
    std::string print();
    std::string print(std::string tensorStr, int dimIndex, int *dimCummulative, int INDEX);
    Tensor<T> operator+(const Tensor<T> &other);

    T *getData()
    {
        return data;
    }
    int *getShape()
    {
        return shape;
    }
    int getDims()
    {
        return dims;
    }
    int getTotalSize()
    {
        return total_size;
    }
};

// Constructor

template <typename T>
Tensor<T>::Tensor(T *data, int *shape, int dims)
{
    std::cout << "3 param constructor was called" << endl;
    this->dims = dims;
    this->shape = new int[this->dims];
    this->total_size = 1;
    for (int i = 0; i < this->dims; ++i)
    {
        this->shape[i] = shape[i];
        this->total_size *= shape[i];
    }
    this->data = new T[this->total_size];
    for (int i = 0; i < this->total_size; ++i)
    {
        this->data[i] = data[i];
    }
}

template <typename T>
Tensor<T>::Tensor(int *shape, int dims)
{
    std::cout << "2 param constructor was called" << endl;
    this->dims = dims;
    this->shape = new int[this->dims];
    this->total_size = 1;
    for (int i = 0; i < this->dims; ++i)
    {
        this->shape[i] = shape[i];
        this->total_size *= shape[i];
    }
    this->data = new T[this->total_size];
}

// Function to initialize Tensor

template <typename T>
Tensor<T> Tensor<T>::getOnes(int *shape, int dims)
{
    Tensor<T> tensor(shape, dims);
    T *data = tensor.getData();
    std::fill(data, data + tensor.getTotalSize(), T(1));
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::getZeroes(int *shape, int dims)
{
    Tensor<T> tensor(shape, dims);
    T *data = tensor.getData();
    std::fill(data, data + tensor.getTotalSize(), T(0));
    return tensor;
}

// Copy Constructor

template <typename T>
Tensor<T>::Tensor(const Tensor &other)
{
    this->dims = other.dims;
    this->total_size = other.total_size;
    this->shape = new int[this->dims];
    this->data = new T[this->total_size];

    for (int i = 0; i < this->dims; ++i)
    {
        this->shape[i] = other.shape[i];
    }

    for (int i = 0; i < this->total_size; ++i)
    {
        this->data[i] = other.data[i];
    }
}

// Desctructor

template <typename T>
Tensor<T>::~Tensor()
{
    delete[] data;
    delete[] shape;
}

// Function to pretty print tensor

template <typename T>
std::string Tensor<T>::print(std::string tensorStr, int dimIndex, int *dimCummulative, int INDEX)
{
    if (dimIndex > this->dims)
    {
        return "";
    }
    else if (this->total_size <= MAX_PRINT_THRESHOLD)
    {
        tensorStr += "[";
        for (int i = 0; i < this->shape[dimIndex]; i++)
        {
            if (dimIndex == this->dims - 1)
            {
                tensorStr += std::to_string(this->data[INDEX + i]);
                if (i != this->shape[dimIndex] - 1)
                {
                    tensorStr += ", ";
                }
            }
            else
            {
                tensorStr = this->print(tensorStr, dimIndex + 1, dimCummulative, INDEX);
                if (i != this->shape[dimIndex] - 1)
                {
                    tensorStr += ",\n";
                }
                INDEX += dimCummulative[dimIndex];
            }
        }
        tensorStr += "]";
        if (dimIndex != this->dims - 1)
        {
            tensorStr += "\n";
        }
    }
    else
    {
        tensorStr += "[";
        bool adddedDot = 0;
        for (int i = 0; i < this->shape[dimIndex]; i++)
        {
            if (dimIndex == this->dims - 1)
            {
                tensorStr += std::to_string(this->data[INDEX + i]);
                if (i != this->shape[dimIndex] - 1)
                {
                    tensorStr += ", ";
                }
                if (this->shape[dimIndex] > MIN_PRINT_THRESHOLD && !adddedDot &&
                    i == (MIN_PRINT_THRESHOLD / 2) - 1)
                {
                    adddedDot = 1;
                    i = this->shape[dimIndex] - 4;
                    tensorStr += "...";
                }
            }
            else
            {
                tensorStr = this->print(tensorStr, dimIndex + 1, dimCummulative, INDEX);
                if (i != this->shape[dimIndex] - 1)
                {
                    tensorStr += ",\n";
                }
                if (this->shape[dimIndex] > MIN_PRINT_THRESHOLD &&
                    i == (MIN_PRINT_THRESHOLD / 2) - 1)
                {
                    i = this->shape[dimIndex] - 4;
                    tensorStr += "\n...\n";
                }
                INDEX += dimCummulative[dimIndex];
            }
        }
        tensorStr += "]";
        if (dimIndex != this->dims - 1)
        {
            tensorStr += "\n";
        }
    }
    return tensorStr;
}
template <typename T>
std::string Tensor<T>::print()
{
    int *dimCummulative = new int[this->dims];
    dimCummulative[this->dims - 1] = 0;
    int aggregate = this->shape[this->dims - 1];
    for (int i = this->dims - 2; i >= 0; --i)
    {
        dimCummulative[i] = aggregate;
        aggregate *= this->shape[i];
    }
    std::string tensorStr = "";
    return print(tensorStr, 0, dimCummulative, 0);
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &other)
{
    if (this->dims != other.dims || this->total_size != other.total_size)
    {
        throw std::invalid_argument("Shape/Size mismatch for tensor addition.");
    }

    for (int i = 0; i < this->dims; ++i)
    {
        if (this->shape[i] != other.shape[i])
        {
            throw std::invalid_argument(
                "Shape mismatch: Tensors must have the same shape to perform first operation.");
        }
    }

    T *device_tensor_A = nullptr;
    T *device_tensor_B = nullptr;
    T *device_tensor_C = nullptr;
    T *host_data = new T[this->total_size];

    cudaError_t cudaStatus;

    // Choosing first GPU to run.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void **)&device_tensor_A, this->total_size * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **)&device_tensor_B, other.total_size * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void **)&device_tensor_C, this->total_size * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("cudaMalloc failed!");
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(device_tensor_A, this->data, this->total_size * sizeof(T),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(device_tensor_B, other.data, other.total_size * sizeof(T),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("cudaMemcpy failed!");
    }

    launchAddKernel<T>(device_tensor_C, device_tensor_A, device_tensor_B, 100);

    cudaStatus = cudaGetLastError();
    std::cout << "Before LastError " << cudaStatus << endl;
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("addKernel launch failed:  ");
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    std::cout << "After sync" << endl;
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(host_data, device_tensor_C, this->total_size * sizeof(T),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("cudaMemcpy failed!");
    }
    cudaFree(device_tensor_A);
    cudaFree(device_tensor_B);
    cudaFree(device_tensor_C);

    return Tensor<T>(host_data, this->shape, this->dims);
}