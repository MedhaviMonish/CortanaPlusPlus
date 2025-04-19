#pragma once
#include "BaseLayer.h"
#include "Tensor.h"
#include "WeightBiasInitializer.h"

template <typename T>
class DenseLayer : public BaseLayer<T>
{
  public:
    DenseLayer<T>(int *shape, int dims);
};
template <typename T>
DenseLayer<T>::DenseLayer(int *shape, int dims)
{
    auto pair = WeightBiasInitializer<T>::initWeightRandom(shape, dims);
    this->weights = pair.first;
    this->bias = pair.second;
}
