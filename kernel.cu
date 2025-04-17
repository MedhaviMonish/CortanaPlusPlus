#include "Tensor.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {4, 15};
    int data_input[60] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                          1, 1, 2, 2, 3, 3, 4, 4, 5, 5,  0,  1, 0, 1, 0, 1, 0, 1, 0, 1,
                          9, 8, 7, 6, 5, 4, 3, 2, 1, 0,  0,  0, 1, 1, 2, 2, 3, 3, 4, 4};
    Tensor<int> array1(data_input, shape_input, 2);

    int shape_weights[] = {8, 15};
    int data_weights[120] = {
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    };
    Tensor<int> array2(data_weights, shape_weights, 2);

    // Perform broadcasted matmul
    Tensor<int> array3 = Tensor<int>::matMul(array1, array2);

    // Print output
    std::cout << array3.print();
    return 0;
}
