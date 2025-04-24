#pragma once
#include "BaseLayer.h"
#include "WeightBiasInitializer.h"
#include <sstream>

template <typename T>
class DenseLayer : public BaseLayer<T>
{
  private:
    int input_size, output_size;

  public:
    DenseLayer<T>(int input_size, int output_size, ACTIVATION activation = ACTIVATION::Linear,
                  INITIALIZATION init = INITIALIZATION::RANDOMS);

    Tensor<T> forward(Tensor<T> &input);
    std::string summary();
};
template <typename T>
DenseLayer<T>::DenseLayer(int input_size, int output_size, ACTIVATION activation,
                          INITIALIZATION init)
{
    // Input is the features which is treated as columns
    // Output is the number of neurons since we follow paper style weights and neurons
    int shape[] = {output_size, input_size};
    int dims = 2;
    std::pair<Tensor<T>, Tensor<T>> pair;

    if (init == INITIALIZATION::ZEROES)
    {
        pair = WeightBiasInitializer<T>::initWeightZeroes(shape, dims);
    }
    else if (init == INITIALIZATION::ONES)
    {
        pair = WeightBiasInitializer<T>::initWeightOnes(shape, dims);
    }
    else if (init == INITIALIZATION::RANDOMS)
    {
        pair = WeightBiasInitializer<T>::initWeightRandom(shape, dims);
    }
    this->weights = pair.first;
    this->bias = pair.second;
    this->activation = activation;
    this->initialization = init;
    this->input_size = input_size;
    this->output_size = output_size;
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

template <typename T>
std::string DenseLayer<T>::summary()
{
    std::ostringstream oss;
    oss << "Dense(" << input_size << " -> " << output_size << "), "
        << "Activation: " << this->activationToString(activation)
        << ", Init: " << this->initToString(initialization) << endl;
    return oss.str();
}
