#include <iostream>
#include <assert.h>

#include "src/tensor.h"
#include "test/test.h"

using namespace jb;
using namespace jb::tensor;
using namespace jb::test;

void TestTensorShapeToStride() {
  {
    vector<int> shape = {4, 2, 3, 5};
    vector<int> strides = tensor::ShapeToStrides(shape);
    AssertTrue(strides[0] == 2 * 3 * 5, "Incorrect stride from shape");
    AssertTrue(strides[1] == 3 * 5, "Incorrect stride from shape");
    AssertTrue(strides[2] == 5, "Incorrect stride from shape");
    AssertTrue(strides[3] == 1, "Incorrect stride from shape");
  }
  {
    vector<int> shape = {5};
    vector<int> strides = tensor::ShapeToStrides(shape);
    AssertTrue(strides[0] == 1, "Incorrect stride from shape");
  }
}

void TestTensorConstructorShapeStride() {
  {
    Tensor<Float32> tensor({2, 3, 9});
    AssertTrue(tensor.Shape()[0] == 2, "Incorrect tensor shape");
    AssertTrue(tensor.Shape()[1] == 3, "Incorrect tensor shape");
    AssertTrue(tensor.Shape()[2] == 9, "Incorrect tensor shape");
    AssertTrue(tensor.Stride()[0] == 3 * 9, "Incorrect tensor stride");
    AssertTrue(tensor.Stride()[1] == 9, "Incorrect tensor stride");
    AssertTrue(tensor.Stride()[2] == 1, "Incorrect tensor stride");
  }
}

int main() {
  TestTensorShapeToStride();
  TestTensorConstructorShapeStride();
  return 0;
}