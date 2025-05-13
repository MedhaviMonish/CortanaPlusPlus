#include "DenseLayer.h"
#include "Loss.h"
#include "SequentialModel.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {2, 6};
    int choice[] = {1, 0};
    Tensor<float> input = Tensor<float>::getOnes(shape_input, 2);
    cout << "Input" << endl;
    cout << input.print() << endl;
    SequentialModel<float> model;
    model.add(new DenseLayer<float>(6, 1, ACTIVATION::Linear, INITIALIZATION::ONES))
        .add(new DenseLayer<float>(1, 2, ACTIVATION::Linear, INITIALIZATION::ONES));

    cout << model.summary();

    Tensor<float> output = model.forward(input);
    cout << "Model Output" << endl;
    int shape[] = {2, 2};
    Tensor<float> target = Tensor<float>::getOnes(shape, 2);
    cout << "output.print() " << endl;
    cout << output.print() << endl;
    cout << "Loss" << endl;
    cout << (Loss<float>::MSE(target, output)).print();

    return 0;
}
