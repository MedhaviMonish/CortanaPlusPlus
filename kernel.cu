#include "Tensor.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{ // ======== Test Input Setup for Broadcasted Elementwise MatMul ========
    int shape_input[] = {2, 6};
    int data_input[12] = {1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12};
    Tensor<int> array1(data_input, shape_input, 2);

    int shape_weights[] = {6, 6};
    int data_weights[36] = {1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2,
                            1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2};
    Tensor<int> array2(data_weights, shape_weights, 2);

    // Perform broadcasted matmul
    Tensor<int> array3 = Tensor<int>::matMul(array1, array2);

    std::cout << array1.print() << endl;
    std::cout << array2.print() << endl;
    std::cout << array3.print() << endl;
    Tensor<int> result = Tensor<int>::reduceSumLastAxis(array3);
    std::cout << result.print();

    return 0;
}
