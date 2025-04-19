#pragma once
#include "Tensor.h"
#include <ctime>
#include <random>
#include <utility>

template <typename T>
class WeightBiasInitializer
{
  public:
    static std::pair<Tensor<T>, Tensor<T>> initWeightZeroes(int *shape, int dims);
    static std::pair<Tensor<T>, Tensor<T>> initWeightOnes(int *shape, int dims);
    static std::pair<Tensor<T>, Tensor<T>> initWeightRandom(int *shape, int dims);
};

template <typename T>
std::pair<Tensor<T>, Tensor<T>> WeightBiasInitializer<T>::initWeightZeroes(int *shape, int dims)
{
    Tensor<T> weight = Tensor<T>::getZeroes(shape, dims);
    int biasDims[] = {1, shape[0]};
    Tensor<T> bias = Tensor<T>::getZeroes(biasDims, 2);
    return {weight, bias};
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> WeightBiasInitializer<T>::initWeightOnes(int *shape, int dims)
{
    Tensor<T> weight = Tensor<T>::getOnes(shape, dims);
    int biasDims[] = {1, shape[0]};
    Tensor<T> bias = Tensor<T>::getOnes(biasDims, 2);
    return {weight, bias};
}

template <typename T>
std::pair<Tensor<T>, Tensor<T>> WeightBiasInitializer<T>::initWeightRandom(int *shape, int dims)
{
    static_assert(std::is_floating_point<T>::value, "T must be a float or double");
    Tensor<T> weight = Tensor<T>::getRandom(shape, dims);
    int biasDims[] = {1, shape[0]};
    Tensor<T> bias = Tensor<T>::getRandom(biasDims, 2);
    return {weight, bias};
}
