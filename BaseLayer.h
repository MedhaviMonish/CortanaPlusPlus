#pragma once
#include "Activation.h"
#include "Tensor.h"

template <typename T>
class BaseLayer
{
  protected:
    Tensor<T> broadcastBias(const Tensor<T> &bias, int batch_size);

  public:
    Tensor<T> weights;
    Tensor<T> bias;
    ACTIVATION activation;
    INITIALIZATION initialization;
    int params;

    BaseLayer() = default;
    virtual Tensor<T> forward(Tensor<T> &input) = 0;
    virtual std::string summary() = 0;
    std::string activationToString(ACTIVATION a);
    std::string initToString(INITIALIZATION i);
    int getParams()
    {
        return params;
    };
};

template <typename T>
Tensor<T> BaseLayer<T>::broadcastBias(const Tensor<T> &bias, int batch_size)
{
    int *newShape = new int[2]{batch_size, bias.getShape()[1]};
    T *data = new T[batch_size * bias.getShape()[1]];

    for (int i = 0; i < batch_size; ++i)
    {
        for (int j = 0; j < bias.getShape()[1]; ++j)
        {
            data[i * bias.getShape()[1] + j] = bias.getData()[j];
        }
    }

    return Tensor<T>(data, newShape, 2);
}

template <typename T>
std::string BaseLayer<T>::activationToString(ACTIVATION a)
{
    switch (a)
    {
    case ACTIVATION::Linear:
        return "Linear";
    case ACTIVATION::ReLU:
        return "ReLU";
    // Add more if needed
    default:
        return "Unknown";
    }
}

template <typename T>
std::string BaseLayer<T>::initToString(INITIALIZATION i)
{
    switch (i)
    {
    case INITIALIZATION::ZEROES:
        return "ZEROES";
    case INITIALIZATION::ONES:
        return "ONES";
    case INITIALIZATION::RANDOMS:
        return "RANDOMS";
    default:
        return "Unknown";
    }
}
