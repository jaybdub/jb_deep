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

void TestTensorSize() {
  {
    Tensor<Float32> tensor({2, 3, 9});
    AssertTrue(tensor.Size() == 2 * 3 * 9, "Incorrect tensor size");
  }
}

void TestTensorAdd() {
  // correct output
  {
    Tensor<Int32> t1({2, 3});
    Tensor<Int32> t2({2, 3});
    t1.Data() = {1, 2, 3, 1, 2, 3};
    t2.Data() = {2, 3, 4, 2, 3, 4};
    Tensor<Int32> out = Add<Int32>(t1, t2);
    AssertTrue(out.Data()[0] == 3, "Invalid add result");
    AssertTrue(out.Data()[1] == 5, "Invalid add result");
    AssertTrue(out.Data()[2] == 7, "Invalid add result");
    AssertTrue(out.Data()[3] == 3, "Invalid add result");
    AssertTrue(out.Data()[4] == 5, "Invalid add result");
    AssertTrue(out.Data()[5] == 7, "Invalid add result");
  }
}

void TestTensorMultiply() {
  // correct output
  {
    Tensor<Int32> t1({2, 3});
    Tensor<Int32> t2({2, 3});
    t1.Data() = {1, 2, 3, 1, 2, 3};
    t2.Data() = {2, 3, 4, 2, 3, 4};
    Tensor<Int32> out = Multiply<Int32>(t1, t2);
    AssertTrue(out.Data()[0] == 2, "Invalid add result");
    AssertTrue(out.Data()[1] == 6, "Invalid add result");
    AssertTrue(out.Data()[2] == 12, "Invalid add result");
    AssertTrue(out.Data()[3] == 2, "Invalid add result");
    AssertTrue(out.Data()[4] == 6, "Invalid add result");
    AssertTrue(out.Data()[5] == 12, "Invalid add result");
  }
}



int main() {
  TestTensorShapeToStride();
  TestTensorConstructorShapeStride();
  TestTensorSize();
  TestTensorAdd();
  TestTensorMultiply();
  return 0;
}