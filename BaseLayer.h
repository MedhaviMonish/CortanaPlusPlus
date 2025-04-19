#pragma once
#include "Tensor.h"
template <typename T>
class BaseLayer
{
  protected:
    Tensor<T> broadcastBias(const Tensor<T> &bias, int batch_size);

  public:
    Tensor<T> weights;
    Tensor<T> bias;

    BaseLayer() = default;
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
