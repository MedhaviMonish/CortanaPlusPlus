#pragma once

template <typename T>
class Tensor
{
private:
    T* data;
    int* shape;
    int dims;
    int total_size;

public:
    Tensor(T* data, int* shape, int dims);
    ~Tensor();

    T* getData() { return data; }
    int* getShape() { return shape; }
    int getDims() { return dims; }
};

template <typename T>
Tensor<T>::Tensor(T* data, int* shape, int dims)
{
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
Tensor<T>::~Tensor() {
    delete[] data;
    delete[] shape;
}
