#pragma once
#include "Activation.h"
#include "Tensor.h"

enum INITIALIZATION
{
    ZEROES = 0,
    ONES = 1,
    RANDOMS = 2,
};

template <typename T>
class BaseLayer
{
  public:
    ACTIVATION activation;
    INITIALIZATION initialization;
    int num_params;

    BaseLayer() = default;
    virtual Tensor<T> forward(Tensor<T> &input) = 0;
    virtual std::string summary() = 0;
    std::string activationToString(ACTIVATION a);
    std::string initToString(INITIALIZATION i);
    int getParams()
    {
        return num_params;
    };
};

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
