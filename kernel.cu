#include "DenseLayer.h"
#include "SequentialModel.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {10, 6};
    Tensor<float> input = Tensor<float>::getRandom(shape_input, 2);
    cout << "Input" << endl;
    cout << input.print() << endl;
    // BaseLayer<float> *layers[];
    SequentialModel<float> model({new DenseLayer<float>(6, 600, ACTIVATION::Linear),
                                  new DenseLayer<float>(600, 20, ACTIVATION::ReLU)});
    Tensor<float> output = model.forward(input);
    cout << "Model Output" << endl;
    cout << output.print() << endl;

    return 0;
}
