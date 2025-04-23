#pragma once
#include "BaseLayer.h"
#include "Tensor.h"
#include <vector>

template <typename T>
class Model
{
  protected:
    std::vector<BaseLayer<T> *> layers;

  public:
    virtual ~Model() = default;
    virtual Tensor<T> forward(const Tensor<T> &input) = 0;
    virtual Model<T> &add(BaseLayer<T> *layer) = 0;
};