#pragma once
#include "TensorKernels.cuh"
#include <ctime>
#include <iostream>
#include <random>
#include <string>

#define MAX_PRINT_THRESHOLD 1000
#define MIN_PRINT_THRESHOLD 6
#define THREADS_PER_BLOCK 256
#define MAX_MEMORY_USAGE_BYTES 256 * 1024 * 1024
// 1024 *1024 *1024  = 1 GB 1073741824

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
    Tensor() : data(nullptr), shape(nullptr), dims(0), total_size(0) {}

    static Tensor<T> getOnes(int *shape, int dims);
    static Tensor<T> getZeroes(int *shape, int dims);
    static Tensor<T> getRandom(int *shape, int dims);
    static Tensor<T> matMul(const Tensor<T> &tensor_A, const Tensor<T> &tensor_B);
    static Tensor<T> reduceSumLastAxis(Tensor<T> &tensor);

    std::string print();
    std::string print(std::string tensorStr, int dimIndex, int *dimCummulative, int INDEX);
    void reshape(int *newShape, int newDims);
    Tensor<T> max(T value);
    Tensor<T> operator+(const Tensor<T> &other);
    Tensor<T> operator+(T value);
    Tensor<T> operator-(const Tensor<T> &other);
    Tensor<T> operator-(T value);
    Tensor<T> operator*(const Tensor<T> &other);
    Tensor<T> operator*(T value);

    Tensor<T> operator=(const Tensor<T> &other);

    T *getData() const
    {
        return data;
    }
    int *getShape() const
    {
        return shape;
    }
    int getDims() const
    {
        return dims;
    }
    int getTotalSize() const
    {
        return total_size;
    }
};

// Constructor

template <typename T>
Tensor<T>::Tensor(T *data, int *shape, int dims)
{
    // std::cout << "3 param constructor was called" << endl;
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
    // std::cout << "3 array" << this->total_size << "  " << this->data[this->total_size - 1] <<
    // endl;
}

template <typename T>
Tensor<T>::Tensor(int *shape, int dims)
{
    // std::cout << "2 param constructor was called" << endl;
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

template <typename T>
Tensor<T> Tensor<T>::getRandom(int *shape, int dims)
{
    static_assert(std::is_floating_point<T>::value, "T must be a float or double");
    Tensor<T> array = Tensor<T>::getZeroes(shape, dims);

    int total = array.getTotalSize();
    T *a_data = array.getData();

    std::default_random_engine engine(static_cast<unsigned>(std::time(0)));
    std::uniform_real_distribution<T> dist(-0.1, 0.1);

    for (int i = 0; i < total; ++i)
        a_data[i] = dist(engine);

    return array;
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
void Tensor<T>::reshape(int *newShape, int newDims)
{
    int tmpTotalSize = 1;
    for (int i = 0; i < newDims; ++i)
    {
        tmpTotalSize *= newShape[i];
    }
    if (tmpTotalSize == this->total_size)
    {
        if (this->shape != nullptr)
        {
            delete[] this->shape;
        }
        this->shape = new int[newDims];
        for (int i = 0; i < newDims; ++i)
        {
            this->shape[i] = newShape[i];
        }
        this->dims = newDims;
    }
    else
    {
        throw std::invalid_argument("New shape has different length than actual data.");
    }
}

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
                    INDEX += ((i - 1) * dimCummulative[dimIndex]);
                    tensorStr += "\n...\n";
                }
                else
                {
                    INDEX += dimCummulative[dimIndex];
                }
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
    std::string tensorStr = print("", 0, dimCummulative, 0);
    delete[] dimCummulative;
    return tensorStr;
}

template <typename T>
Tensor<T> Tensor<T>::operator=(const Tensor<T> &other)
{
    if (this != &other)
    {
        delete[] this->shape;
        delete[] this->data;

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
    return *this;
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
    T *host_data = new T[this->total_size];

    cudaError_t cudaStatus;

    // Choosing first GPU to run.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
    int i = 0;
    while (i < this->total_size)
    {
        if (i + SUB_TOTAL_SIZE > this->total_size)
        {
            SUB_TOTAL_SIZE = this->total_size - i;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void **)&device_tensor_B, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        std::cout << "CudaStatus " << cudaStatus << endl;
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        cudaStatus = cudaMemcpy(device_tensor_B, &other.data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        dim3 thread_per_blocks(THREADS_PER_BLOCK);
        dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

        launchAddKernel<T>(device_tensor_A, device_tensor_B, thread_blocks, thread_per_blocks,
                           SUB_TOTAL_SIZE);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("addKernel launch failed:  ");
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();

        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument(
                "cudaDeviceSynchronize returned error after launching addKernel!");
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(&host_data[i], device_tensor_A, SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }
        i += SUB_TOTAL_SIZE;
        cudaFree(device_tensor_A);
        cudaFree(device_tensor_B);
    }

    return Tensor<T>(host_data, this->shape, this->dims);
}

template <typename T>
Tensor<T> Tensor<T>::operator+(T value)
{
    T *device_tensor_A = nullptr;
    T *host_scalar = new T[this->total_size];
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
    // std::cout << "MAX Subtotal size at start based on data type " << SUB_TOTAL_SIZE << endl;
    int i = 0;
    while (i < this->total_size)
    {
        if (i + SUB_TOTAL_SIZE > this->total_size)
        {
            SUB_TOTAL_SIZE = this->total_size - i;
        }
        // std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;

        cudaStatus = cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        dim3 thread_per_blocks(THREADS_PER_BLOCK);
        dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

        launchAddScalarKernel<T>(device_tensor_A, thread_blocks, thread_per_blocks, value,
                                 SUB_TOTAL_SIZE);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("addKernel launch failed:  ");
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument(
                "cudaDeviceSynchronize returned error after launching addKernel!");
        }

        cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor_A, SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }
        i += SUB_TOTAL_SIZE;
        cudaFree(device_tensor_A);
    }
    int *newShape = new int[this->dims];
    std::memcpy(newShape, this->shape, this->dims * sizeof(int));
    return Tensor<T>(host_scalar, newShape, this->dims);
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &other)
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
    T *host_data = new T[this->total_size];

    cudaError_t cudaStatus;

    // Choosing first GPU to run.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
    int i = 0;
    while (i < this->total_size)
    {
        if (i + SUB_TOTAL_SIZE > this->total_size)
        {
            SUB_TOTAL_SIZE = this->total_size - i;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void **)&device_tensor_B, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        std::cout << "CudaStatus " << cudaStatus << endl;
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        cudaStatus = cudaMemcpy(device_tensor_B, &other.data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        dim3 thread_per_blocks(THREADS_PER_BLOCK);
        dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

        launchSubKernel<T>(device_tensor_A, device_tensor_B, thread_blocks, thread_per_blocks,
                           SUB_TOTAL_SIZE);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("addKernel launch failed:  ");
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();

        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument(
                "cudaDeviceSynchronize returned error after launching addKernel!");
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(&host_data[i], device_tensor_A, SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }
        i += SUB_TOTAL_SIZE;
        cudaFree(device_tensor_A);
        cudaFree(device_tensor_B);
    }

    return Tensor<T>(host_data, this->shape, this->dims);
}

template <typename T>
Tensor<T> Tensor<T>::operator-(T value)
{
    T *device_tensor_A = nullptr;
    T *host_scalar = new T[this->total_size];
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
    // std::cout << "MAX Subtotal size at start based on data type " << SUB_TOTAL_SIZE << endl;
    int i = 0;
    while (i < this->total_size)
    {
        if (i + SUB_TOTAL_SIZE > this->total_size)
        {
            SUB_TOTAL_SIZE = this->total_size - i;
        }
        // std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;

        cudaStatus = cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        dim3 thread_per_blocks(THREADS_PER_BLOCK);
        dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

        launchSubScalarKernel<T>(device_tensor_A, thread_blocks, thread_per_blocks, value,
                                 SUB_TOTAL_SIZE);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("addKernel launch failed:  ");
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument(
                "cudaDeviceSynchronize returned error after launching addKernel!");
        }

        cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor_A, SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }
        i += SUB_TOTAL_SIZE;
        cudaFree(device_tensor_A);
    }
    int *newShape = new int[this->dims];
    std::memcpy(newShape, this->shape, this->dims * sizeof(int));
    return Tensor<T>(host_scalar, newShape, this->dims);
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &other)
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
    T *host_data = new T[this->total_size];

    cudaError_t cudaStatus;

    // Choosing first GPU to run.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
    int i = 0;
    while (i < this->total_size)
    {
        if (i + SUB_TOTAL_SIZE > this->total_size)
        {
            SUB_TOTAL_SIZE = this->total_size - i;
        }

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }

        cudaStatus = cudaMalloc((void **)&device_tensor_B, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        std::cout << "CudaStatus " << cudaStatus << endl;
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        cudaStatus = cudaMemcpy(device_tensor_B, &other.data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        dim3 thread_per_blocks(THREADS_PER_BLOCK);
        dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

        launchMultiplyKernel<T>(device_tensor_A, device_tensor_B, thread_blocks, thread_per_blocks,
                                SUB_TOTAL_SIZE);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("addKernel launch failed:  ");
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();

        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument(
                "cudaDeviceSynchronize returned error after launching addKernel!");
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(&host_data[i], device_tensor_A, SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }
        i += SUB_TOTAL_SIZE;
        cudaFree(device_tensor_A);
        cudaFree(device_tensor_B);
    }

    return Tensor<T>(host_data, this->shape, this->dims);
}

template <typename T>
Tensor<T> Tensor<T>::operator*(T value)
{
    T *device_tensor_A = nullptr;
    T *host_scalar = new T[this->total_size];
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
    // std::cout << "MAX Subtotal size at start based on data type " << SUB_TOTAL_SIZE << endl;
    int i = 0;
    while (i < this->total_size)
    {
        if (i + SUB_TOTAL_SIZE > this->total_size)
        {
            SUB_TOTAL_SIZE = this->total_size - i;
        }
        // std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;

        cudaStatus = cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        dim3 thread_per_blocks(THREADS_PER_BLOCK);
        dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

        launchMultiplyScalarKernel<T>(device_tensor_A, thread_blocks, thread_per_blocks, value,
                                      SUB_TOTAL_SIZE);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("addKernel launch failed:  ");
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument(
                "cudaDeviceSynchronize returned error after launching addKernel!");
        }

        cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor_A, SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }
        i += SUB_TOTAL_SIZE;
        cudaFree(device_tensor_A);
    }
    int *newShape = new int[this->dims];
    std::memcpy(newShape, this->shape, this->dims * sizeof(int));
    return Tensor<T>(host_scalar, newShape, this->dims);
}

template <typename T>
Tensor<T> Tensor<T>::matMul(const Tensor<T> &tensor_A, const Tensor<T> &tensor_B)
{

    if (tensor_A.dims != tensor_B.dims && tensor_B.dims != 2)
    {
        throw std::invalid_argument("Shape/Size mismatch for tensor addition.");
    }
    if (tensor_A.shape[tensor_A.dims - 1] != tensor_B.shape[tensor_B.dims - 1])
    {
        throw std::invalid_argument("First and Last dimension mismatch for Tensor MatMul.");
    }

    T *device_tensor_A = nullptr;
    T *device_tensor_B = nullptr;
    T *device_tensor_Mul = nullptr;

    T *host_matmul_data = new T[tensor_A.shape[0] * tensor_B.total_size];

    cudaError_t cudaStatus;

    // Choosing first GPU to run.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    cudaStatus = cudaMalloc((void **)&device_tensor_A, tensor_A.total_size * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("Memory allocation failed for tensor_A");
    }

    cudaStatus = cudaMalloc((void **)&device_tensor_B, tensor_B.total_size * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("Memory allocation failed for tensor_B");
    }

    cudaStatus = cudaMalloc((void **)&device_tensor_Mul,
                            tensor_A.shape[0] * tensor_B.total_size * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("Memory allocation failed for device_tensor_Mul");
    }

    cudaStatus = cudaMemcpy(device_tensor_A, tensor_A.data, tensor_A.total_size * sizeof(T),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        throw std ::invalid_argument("Data copy failed for tensor_A");
    }

    cudaStatus = cudaMemcpy(device_tensor_B, tensor_B.data, tensor_B.total_size * sizeof(T),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        throw std ::invalid_argument("Data copy failed for tensor_B");
    }

    dim3 thread_per_blocks(tensor_A.shape[0]);
    dim3 thread_blocks(tensor_B.shape[0], tensor_B.shape[1]);
    launchMatMulKernel<T>(device_tensor_A, device_tensor_B, device_tensor_Mul, thread_blocks,
                          thread_per_blocks);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("addKernel launch failed:  ");
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered
    // during the launch.
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus =
        cudaMemcpy(host_matmul_data, device_tensor_Mul,
                   tensor_A.shape[0] * tensor_B.total_size * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("cudaMemcpy failed!");
    }

    cudaFree(device_tensor_A);
    cudaFree(device_tensor_B);
    cudaFree(device_tensor_Mul);

    int shape[] = {tensor_A.shape[0], tensor_B.shape[0], tensor_B.shape[1]};
    return Tensor<T>(host_matmul_data, shape, 3);
}

template <typename T>
Tensor<T> Tensor<T>::reduceSumLastAxis(Tensor<T> &tensor)
{
    if (tensor.dims == 1)
    {
        // We simply add extra dim if it has only one dimension
        int tmp[] = {1, tensor.shape[0]};
        tensor.reshape(tmp, 2);
    }
    int total_size = tensor.total_size;
    int dimCummulative = 1;
    int *newShape = new int[tensor.dims - 1];
    for (int i = 0; i < tensor.dims - 1; ++i)
    {
        dimCummulative *= tensor.shape[i];
        newShape[i] = tensor.shape[i];
    }

    int currentLastShape = tensor.shape[tensor.dims - 1];
    int THREADS = 4;
    int STRIDE = 2;

    if (((currentLastShape / STRIDE) + 1) > 1024)
    {
        THREADS = THREADS; // If based on this stride, threads are needed moore than 1024 then limit
                           // it to specified number as it is
        STRIDE = currentLastShape / THREADS;
    }
    else
    {
        // Now its less than 1024. lets check if we really need this much
        // Stride should be greater than 256
        if (((currentLastShape / THREADS) + 1) < 256)
        {
            STRIDE = 256;
            THREADS = (currentLastShape / STRIDE) + 1;
        }
    }

    int reducedLastDimShape = THREADS;
    T *reducedSum = new T[dimCummulative];
    T *deviceTensor = nullptr;
    T *deviceReducedSum = nullptr;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    cudaStatus = cudaMalloc((void **)&deviceTensor, total_size * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("Memory allocation failed for tensor");
    }

    cudaStatus =
        cudaMalloc((void **)&deviceReducedSum, dimCummulative * reducedLastDimShape * sizeof(T));
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("Memory allocation failed for tensor");
    }

    cudaStatus = cudaMemcpy(deviceTensor, tensor.data, tensor.total_size * sizeof(T),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        throw std ::invalid_argument("Data copy failed for tensor_A");
    }

    dim3 thread_per_blocks(THREADS);
    dim3 thread_blocks(dimCummulative);
    launchReduceSumLastAxisKernel<T>(deviceTensor, deviceReducedSum, thread_blocks,
                                     thread_per_blocks, STRIDE, currentLastShape);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("addKernel launch failed:  ");
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered
    // during the launch.
    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaDeviceSynchronize returned error after launching addKernel!");
    }

    cudaFree(deviceTensor);

    while (reducedLastDimShape > 1)
    {
        currentLastShape = reducedLastDimShape;
        THREADS = 4;
        STRIDE = 2;

        // Using the last Dim of intermediate sum results
        if (((reducedLastDimShape / STRIDE) + 1) > 1024)
        {
            THREADS = THREADS; // If based on this stride, threads are needed moore than 1024 then
                               // limit it to specified number as it is
            STRIDE = reducedLastDimShape / THREADS;
        }
        else
        {
            // Now its less than 1024. lets check if we really need this much
            // Stride should be greater than 256
            if (((reducedLastDimShape / THREADS) + 1) < 256)
            {
                STRIDE = 256;
                THREADS = (reducedLastDimShape / STRIDE) + 1;
            }
        }

        reducedLastDimShape = THREADS;
        deviceTensor = deviceReducedSum;

        cudaStatus = cudaMalloc((void **)&deviceReducedSum,
                                dimCummulative * reducedLastDimShape * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("Memory allocation failed for tensor");
        }

        dim3 thread_per_blocks(THREADS);
        dim3 thread_blocks(dimCummulative);
        launchReduceSumLastAxisKernel<T>(deviceTensor, deviceReducedSum, thread_blocks,
                                         thread_per_blocks, STRIDE, currentLastShape);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("addKernel launch failed:  ");
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered
        // during the launch.
        cudaStatus = cudaDeviceSynchronize();

        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument(
                "cudaDeviceSynchronize returned error after launching addKernel!");
        }

        cudaFree(deviceTensor);
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(reducedSum, deviceReducedSum, dimCummulative * sizeof(T),
                            cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument("cudaMemcpy failed!");
    }
    Tensor<T> result = Tensor<T>(reducedSum, newShape, tensor.dims - 1);

    cudaFree(deviceReducedSum);
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::max(T value)
{
    T *device_tensor_A = nullptr;
    T *host_scalar = new T[this->total_size];
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        throw std::invalid_argument(
            "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    int SUB_TOTAL_SIZE = (MAX_MEMORY_USAGE_BYTES) / sizeof(T);
    // std::cout << "MAX Subtotal size at start based on data type " << SUB_TOTAL_SIZE << endl;
    int i = 0;
    while (i < this->total_size)
    {
        if (i + SUB_TOTAL_SIZE > this->total_size)
        {
            SUB_TOTAL_SIZE = this->total_size - i;
        }
        // std::cout << "Subtotal size " << SUB_TOTAL_SIZE << endl;

        cudaStatus = cudaMalloc((void **)&device_tensor_A, SUB_TOTAL_SIZE * sizeof(T));
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMalloc failed!");
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(device_tensor_A, &this->data[i], SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }

        dim3 thread_per_blocks(THREADS_PER_BLOCK);
        dim3 thread_blocks((SUB_TOTAL_SIZE / THREADS_PER_BLOCK) + 1);

        launchMaxScalarKernel<T>(device_tensor_A, thread_blocks, thread_per_blocks, value,
                                 SUB_TOTAL_SIZE);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("maxScalarKernel launch failed:  ");
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument(
                "cudaDeviceSynchronize returned error after launching maxScalarKernel!");
        }

        cudaStatus = cudaMemcpy(&host_scalar[i], device_tensor_A, SUB_TOTAL_SIZE * sizeof(T),
                                cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess)
        {
            throw std::invalid_argument("cudaMemcpy failed!");
        }
        i += SUB_TOTAL_SIZE;
        cudaFree(device_tensor_A);
    }
    int *newShape = new int[this->dims];
    std::memcpy(newShape, this->shape, this->dims * sizeof(int));
    return Tensor<T>(host_scalar, newShape, this->dims);
}
