#pragma once
#include "Tensor.h"
#include <utility>

template <typename T>
class WeightBiasInitializer
{
  public:
    static std::pair<Tensor<T>, Tensor<T>> initWeightZeroes(int *shape, int dims);
    static std::pair<Tensor<T>, Tensor<T>> initWeightOnes(int *shape, int dims);
};

template <typename T>
std::pair<Tensor<T>, Tensor<T>> WeightBiasInitializer<T>::initWeightZeroes(int *shape, int dims)
{
    Tensor<T> weight = Tensor<T>::getZeroes(shape, dims);
    int biasDims[] = {shape[0], 1};
    Tensor<T> bias = Tensor<T>::getZeroes(biasDims, 1);
    return {weight, bias};
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> WeightBiasInitializer<T>::initWeightOnes(int *shape, int dims)
{
    Tensor<T> weight = Tensor<T>::getOnes(shape, dims);
    int biasDims[] = {shape[0], 1};
    Tensor<T> bias = Tensor<T>::getOnes(biasDims, 1);
    return {weight, bias};
}
