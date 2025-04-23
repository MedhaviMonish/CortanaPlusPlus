#pragma once

#include "BaseLayer.h"
#include "Model.h"

template <typename T>
class SequentialModel : public Model<T>
{
  protected:
    std::vector<BaseLayer<T> *> layers;

  public:
    SequentialModel() = default;
    SequentialModel(std::initializer_list<BaseLayer<T> *> layer_array);

    Tensor<T> forward(const Tensor<T> &input);
    void add(BaseLayer<T> &layer);
};

template <typename T>
SequentialModel<T>::SequentialModel(std::initializer_list<BaseLayer<T> *> layer_array)
{
    for (BaseLayer<T> *layer : layer_array)
        this->layers.push_back(layer);
}

template <typename T>
Tensor<T> SequentialModel<T>::forward(const Tensor<T> &input)
{
    Tensor<T> out = input;
    for (auto *layer : this->layers)
    {
        // assumes BaseLayer<T> has virtual Tensor<T> forward(const Tensor<T>&)
        out = layer->forward(out);
    }
    return out;
}

template <typename T>
void SequentialModel<T>::add(BaseLayer<T> &layer)
{
    this->layers.push_back(&layer);
}
