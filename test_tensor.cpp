// #define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
// #include "Tensor.h"
// #include "doctest.h"
//
// TEST_CASE("Tensor creation from data")
//{
//     int data[] = {1, 2, 3, 4, 5, 6};
//     int shape[] = {6};
//
//     Tensor<int> tensor(data, shape, 1);
//
//     CHECK(tensor.getDims() == 1);
//     CHECK(tensor.getTotalSize() == 6);
//
//     int *tensorData = tensor.getData();
//     for (int i = 0; i < 6; ++i)
//     {
//         CHECK(tensorData[i] == data[i]);
//     }
// }
//
//// TEST_CASE("Tensor creation with getOnes")
////{
////     int shape[] = {5};
////     Tensor<float> tensor = Tensor<float>::getOnes(shape, 1);
////     std::cout << "Dims = " << tensor.getDims() << "\n";
////     std::cout << "Total size = " << tensor.getTotalSize() << "\n";
////     float *data = tensor.getData();
////     for (int i = 0; i < tensor.getTotalSize(); ++i)
////     {
////         std::cout << "data[" << i << "] = " << data[i] << "\n";
////     }
////
////     CHECK(tensor.getDims() == 1);
////     CHECK(tensor.getTotalSize() == 5);
////
////     float *tensorData = tensor.getData();
////     for (int i = 0; i < 5; ++i)
////     {
////         CHECK(tensorData[i] == 1.0f);
////     }
//// }
//
// TEST_CASE("Tensor creation with getZeroes")
//{
//    int shape[] = {3};
//    Tensor<int> tensor = Tensor<int>::getZeroes(shape, 1);
//
//    CHECK(tensor.getDims() == 1);
//    CHECK(tensor.getTotalSize() == 3);
//
//    int *tensorData = tensor.getData();
//    for (int i = 0; i < 3; ++i)
//    {
//        CHECK(tensorData[i] == 0);
//    }
//}
