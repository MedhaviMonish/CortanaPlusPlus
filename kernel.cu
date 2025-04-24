#include "DenseLayer.h"
#include "Loss.h"
#include "SequentialModel.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {1, 6};
    int choice[] = {1, 0};
    Tensor<float> input = Tensor<float>::getRandom(shape_input, 2, choice, 2);
    cout << "Input" << endl;
    cout << input.print() << endl;
    SequentialModel<float> model;
    model.add(new DenseLayer<float>(6, 1, ACTIVATION::Linear))
        .add(new DenseLayer<float>(1, 6, ACTIVATION::Linear, INITIALIZATION::ONES));

    std::vector<BaseLayer<float> *> layers = model.getLayers();
    cout << layers[0]->summary();

    Tensor<float> output = model.forward(input);
    cout << "Model Output" << endl;
    int shape[] = {6, 1};
    output.reshape(shape, 2);
    input.reshape(shape, 2);
    cout << output.print() << endl;
    cout << "Input" << endl;
    cout << input.print() << endl;
    cout << (Loss<float>::MSE(input, output)).print();

    return 0;
}
