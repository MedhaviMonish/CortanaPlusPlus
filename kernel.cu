#include "DenseLayer.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {2, 6};
    DenseLayer<float> layer = DenseLayer<float>(shape_input, 2);
    cout << layer.weights.print();
    cout << layer.bias.print();

    return 0;
}
