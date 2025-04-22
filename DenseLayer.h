#pragma once
#include "Activation.h"
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
    ACTIVATION activation;
    DenseLayer<T>(int input_size, int output_size, ACTIVATION activation = ACTIVATION::Linear,
                  Initialization init = Initialization::RANDOMS);

    Tensor<T> forward(Tensor<T> &input);
};
template <typename T>
DenseLayer<T>::DenseLayer(int input_size, int output_size, ACTIVATION activation,
                          Initialization init)
{
    // Input is the features which is treated as columns
    // Output is the number of neurons since we follow paper style weights and neurons
    int shape[] = {output_size, input_size};
    int dims = 2;
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
    this->activation = activation;
}

template <typename T>
Tensor<T> DenseLayer<T>::forward(Tensor<T> &input)
{
    Tensor<T> output = Tensor<T>::matMul(input, this->weights);
    output = Tensor<T>::reduceSumLastAxis(output);
    Tensor<T> tmp_bias = this->broadcastBias(this->bias, input.getShape()[0]);
    output = output + tmp_bias;
    if (this->activation == ACTIVATION::ReLU)
    {
        return Activation<T>::ReLU(output);
    }
    else
    {
        return output;
    }
}
