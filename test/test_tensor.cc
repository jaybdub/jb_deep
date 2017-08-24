#include <iostream>
#include <assert.h>

#include "src/tensor.h"
#include "test/test.h"

using namespace jb;
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

int main() {
  TestTensorShapeToStride();
  return 0;
}