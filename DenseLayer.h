#pragma once
#include "BaseLayer.h"
#include "Tensor.h"
#include "WeightBiasInitializer.h"

enum Initialization
{
    ZEROES = 0,
    ONES = 1,
    RANDOMS = 2,
};

template <typename T>
class DenseLayer : public BaseLayer<T>
{
  public:
    DenseLayer<T>(int *shape, int dims, Initialization init = Initialization::RANDOMS);
    Tensor<T> forward(Tensor<T> &input);
};
template <typename T>
DenseLayer<T>::DenseLayer(int *shape, int dims, Initialization init)
{
    std::pair<Tensor<T>, Tensor<T>> pair;

    if (init == Initialization::ZEROES)
    {
        pair = WeightBiasInitializer<T>::initWeightZeroes(shape, dims);
    }
    else if (init == Initialization::ONES)
    {
        pair = WeightBiasInitializer<T>::initWeightOnes(shape, dims);
    }
    else if (init == Initialization::RANDOMS)
    {
        pair = WeightBiasInitializer<T>::initWeightRandom(shape, dims);
    }
    this->weights = pair.first;
    this->bias = pair.second;
}

template <typename T>
Tensor<T> DenseLayer<T>::forward(Tensor<T> &input)
{
    Tensor<T> output = Tensor<T>::matMul(input, this->weights);
    output = Tensor<T>::reduceSumLastAxis(output);
    return output;
}
