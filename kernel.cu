#include "Tensor.h"
#include <iostream>
#include <stdio.h>
#include <string>
using namespace std;

int main()
{
    int shape[] = {5000, 200, 500};
    int dims = 3;
    int *array = nullptr;
    int total = 1;
    for (int i = 0; i < dims; ++i)
    {
        total *= shape[i];
    }
    array = new int[total];
    for (int i = 0; i < total; ++i)
    {
        array[i] = i + 1;
    }
    Tensor<int> array1 = Tensor<int>(array, shape, dims);
    cout << array1.print();
    cout << array1.getData()[array1.getTotalSize() - 1];

    Tensor<int> array2 = Tensor<int>::getOnes(shape, dims);
    cout << "Print" << endl;
    cout << array1.getTotalSize() << endl;
    Tensor<int> array3 = array1 + array2;
    cout << array3.print();

    return 0;
}
