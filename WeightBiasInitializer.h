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

template <typename T>
std::pair<Tensor<T>, Tensor<T>> WeightBiasInitializer<T>::initWeightRandom(int *shape, int dims)
{
    static_assert(std::is_floating_point<T>::value, "T must be a float or double");
    Tensor<T> weight = Tensor<T>::getZeroes(shape, dims);
    int biasDims[] = {shape[0], 1};
    Tensor<T> bias = Tensor<T>::getZeroes(biasDims, 1);

    int total = weight.getTotalSize();
    T *w_data = weight.getData();
    T *b_data = bias.getData();

    std::default_random_engine engine(static_cast<unsigned>(std::time(0)));
    std::uniform_real_distribution<T> dist(-0.1, 0.1);

    for (int i = 0; i < total; ++i)
        w_data[i] = dist(engine);

    for (int i = 0; i < shape[0]; ++i)
        b_data[i] = dist(engine);

    return {weight, bias};
}
