#pragma once

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
