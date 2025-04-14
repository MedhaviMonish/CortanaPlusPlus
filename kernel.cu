#include "Tensor.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{
    int shape[] = {10, 5};
    int dims = 2;
    Tensor<int> array1 = Tensor<int>::getOnes(shape, dims);
    int shape1[] = {4, 5};
    Tensor<int> array2 = Tensor<int>::getOnes(shape1, dims);
    cout << array1.print() << endl;
    array2 = array2 + 2;
    cout << "array2 After scalar addition" << endl;
    cout << array2.print() << endl;

    Tensor<int> array4 = Tensor<int>::matMul(array1, array2);
    cout << array4.print();
    return 0;
}
