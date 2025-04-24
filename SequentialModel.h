#pragma once

#include "BaseLayer.h"
#include "Model.h"

template <typename T>
class SequentialModel : public Model<T>
{

  public:
    SequentialModel() = default;
    SequentialModel(std::initializer_list<BaseLayer<T> *> layer_array);

    Tensor<T> forward(const Tensor<T> &input);
    SequentialModel<T> &add(BaseLayer<T> *layer);
    std::vector<BaseLayer<T> *> getLayers();
    std::string summary();
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
    for (BaseLayer<T> *layer : this->layers)
    {
        // assumes BaseLayer<T> has virtual Tensor<T> forward(const Tensor<T>&)
        out = layer->forward(out);
    }
    return out;
}

template <typename T>
SequentialModel<T> &SequentialModel<T>::add(BaseLayer<T> *layer)
{
    this->layers.push_back(layer);
    return *this;
}

template <typename T>
std::vector<BaseLayer<T> *> SequentialModel<T>::getLayers()
{
    return this->layers;
}

template <typename T>
std::string SequentialModel<T>::summary()
{
    std::ostringstream oss;
    int params = 0;
    oss << "===========================================================" << endl;
    oss << "Total Layers " << this->layers.size() << endl;
    oss << "___________________________________________________________" << endl;
    for (BaseLayer<T> *layer : this->layers)
    {
        oss << layer->summary();
        params += layer->getParams();
    }
    oss << "___________________________________________________________" << endl;
    oss << "Total Parameters " << params << endl;
    oss << "===========================================================" << endl;
    return oss.str();
}