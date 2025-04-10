#pragma once
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
    std::string print(std::string tensorStr, int *dimCummulative, int dimIndex);

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
Tensor<T>::~Tensor()
{
    delete[] data;
    delete[] shape;
}

template <typename T>
std::string Tensor<T>::print(std::string tensorStr, int *dimCummulative, int dimIndex)
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
                tensorStr += std::to_string(this->data[i]);
                if (i != this->shape[dimIndex] - 1)
                {
                    tensorStr += ", ";
                }
            }
            else
            {
                tensorStr = this->print(tensorStr, dimCummulative, dimIndex + 1);
                if (i != this->shape[dimIndex] - 1)
                {
                    tensorStr += ",\n";
                }
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
        for (int i = 0; i < this->shape[dimIndex]; i++)
        {
            if (dimIndex == this->dims - 1)
            {
                tensorStr += std::to_string(this->data[i]);
                if (i != this->shape[dimIndex] - 1)
                {
                    tensorStr += ", ";
                }
                if (i == (MIN_PRINT_THRESHOLD / 2) - 1)
                {
                    i = this->shape[dimIndex] - 4;
                    tensorStr += "...";
                }
            }
            else
            {
                tensorStr = this->print(tensorStr, dimCummulative, dimIndex + 1);
                if (i != this->shape[dimIndex] - 1)
                {
                    tensorStr += ",\n";
                }
                if (i == (MIN_PRINT_THRESHOLD / 2) - 1)
                {
                    i = this->shape[dimIndex] - 4;
                    tensorStr += "\n...\n";
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
    int aggregate = 1;
    for (int i = this->dims - 1; i >= 0; --i)
    {
        aggregate *= this->shape[i];
        dimCummulative[i] = aggregate;
    }
    std::string tensorStr = "";
    return print(tensorStr, dimCummulative, 0);
}