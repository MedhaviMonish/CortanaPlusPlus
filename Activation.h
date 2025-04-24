#pragma once
#include "Tensor.h"

enum INITIALIZATION
{
    ZEROES = 0,
    ONES = 1,
    RANDOMS = 2,
};

enum ACTIVATION
{
    Linear = 0,
    ReLU = 1,
};

template <typename T>
class Activation
{
  public:
    static Tensor<T> ReLU(Tensor<T> &tensor);
};

template <typename T>
Tensor<T> Activation<T>::ReLU(Tensor<T> &tensor)
{
    T value = 0;
    return tensor.max(value);
}
