#pragma once
#include "Tensor.h"
template <typename T>
class BaseLayer
{
  public:
    Tensor<T> weights;
    Tensor<T> bias;

    BaseLayer() = default;
};
