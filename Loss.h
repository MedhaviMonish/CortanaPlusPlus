#pragma once
#include "Tensor.h"

template <typename T>
class Loss
{
  public:
    static Tensor<T> MSE(Tensor<T> &target_y, Tensor<T> &prediction_y);
};

template <typename T>
Tensor<T> Loss<T>::MSE(Tensor<T> &target_y, Tensor<T> &prediction_y)
{
    return (target_y - prediction_y).pow(2);
}