#include <iostream>
#include <functional>


#include "src/tensor.h"
#include "test/test.h"

using namespace std;
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
    Tensor<Int32> tensor = Zeros<Int32>({2, 3, 9});
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
    Tensor<Int32> tensor = Zeros<Int32>({2, 3, 9});
    AssertTrue(tensor.Size() == 2 * 3 * 9, "Incorrect tensor size");
  }
}

void TestTensorAdd() {
  // correct output
  {
    Tensor<Int32> t1 = Zeros<Int32>({2, 3});
    Tensor<Int32> t2 = Zeros<Int32>({2, 3});
    t1.DataMutable() = {1, 2, 3, 1, 2, 3};
    t2.DataMutable() = {2, 3, 4, 2, 3, 4};
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
    Tensor<Int32> t1 = Zeros<Int32>({2, 3});
    Tensor<Int32> t2 = Zeros<Int32>({2, 3});
    t1.DataMutable() = {1, 2, 3, 1, 2, 3};
    t2.DataMutable() = {2, 3, 4, 2, 3, 4};
    Tensor<Int32> out = Multiply<Int32>(t1, t2);
    AssertTrue(out.Data()[0] == 2, "Invalid add result");
    AssertTrue(out.Data()[1] == 6, "Invalid add result");
    AssertTrue(out.Data()[2] == 12, "Invalid add result");
    AssertTrue(out.Data()[3] == 2, "Invalid add result");
    AssertTrue(out.Data()[4] == 6, "Invalid add result");
    AssertTrue(out.Data()[5] == 12, "Invalid add result");
  }
}

void TestTensorSubtract() {
  {
    Tensor<Int32> a = Zeros<Int32>({3});
    Tensor<Int32> b = Zeros<Int32>({3});
    a.DataMutable() = {1, 2, 3};
    b.DataMutable() = {2, 4, 6};
    auto c = Subtract(a, b);
    AssertTrue(c.Data()[0] == -1, "Invalid subtract result");
    AssertTrue(c.Data()[1] == -2, "Invalid subtract result");
    AssertTrue(c.Data()[2] == -3, "Invalid subtract result");
  }
}

void TestTensorNegate() {
  {
    Tensor<Int32> a = Zeros<Int32>({3});
    a.DataMutable() = {1, 2, 3};
    auto c = Negate(a);
    AssertTrue(c.Data()[0] == -1, "Invalid negate result");
    AssertTrue(c.Data()[1] == -2, "Invalid negate result");
    AssertTrue(c.Data()[2] == -3, "Invalid negate result");
  }
}

void TestTensorApply() {
  {
    Tensor<Int32> a = Zeros<Int32>({3});
    a.DataMutable() = {-1, -2, -3};
    auto c = Apply(a, std::abs);
    AssertTrue(c.Data()[0] == 1, "Invalid apply result");
    AssertTrue(c.Data()[1] == 2, "Invalid apply result");
    AssertTrue(c.Data()[2] == 3, "Invalid apply result");
  }
}

void TestTensorGet() {
  {
    Tensor<Int32> a = Zeros<Int32>({3, 3});
    a.DataMutable() = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    AssertTrue(a.Get({0, 0}) == 1, "Invalid get result");
    AssertTrue(a.Get({0, 1}) == 2, "Invalid get result");
    AssertTrue(a.Get({0, 2}) == 3, "Invalid get result");
    AssertTrue(a.Get({1, 0}) == 4, "Invalid get result");
    AssertTrue(a.Get({1, 1}) == 5, "Invalid get result");
    AssertTrue(a.Get({1, 2}) == 6, "Invalid get result");
    AssertTrue(a.Get({2, 0}) == 7, "Invalid get result");
    AssertTrue(a.Get({2, 1}) == 8, "Invalid get result");
    AssertTrue(a.Get({2, 2}) == 9, "Invalid get result");
  }
}

void TestTensorAt() {
  {
    Tensor<Int32> a = Zeros<Int32>({3, 3});
    a.DataMutable() = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    AssertTrue(a.At({0, 0}) == 1, "Invalid at result");
    AssertTrue(a.At({0, 1}) == 2, "Invalid at result");
    AssertTrue(a.At({0, 2}) == 3, "Invalid at result");
    AssertTrue(a.At({1, 0}) == 4, "Invalid at result");
    AssertTrue(a.At({1, 1}) == 5, "Invalid at result");
    AssertTrue(a.At({1, 2}) == 6, "Invalid at result");
    AssertTrue(a.At({2, 0}) == 7, "Invalid at result");
    AssertTrue(a.At({2, 1}) == 8, "Invalid at result");
    AssertTrue(a.At({2, 2}) == 9, "Invalid at result");
    // modify element
    a.At({2, 2}) = 15;
    AssertTrue(a.At({2, 2}) == 15, "Should modify element returned by At()");
  }
}

void TestTensorMatrixMultply() {
  {
    Tensor<Int32> a = Zeros<Int32>({3, 2});
    Tensor<Int32> b = Zeros<Int32>({2, 3});
    a.DataMutable() = {1, 2, 1, 2, 1, 2};
    b.DataMutable() = {1, 2, 3, 1, 2, 3};
    auto c = MatrixMultiply(a, b);
    AssertTrue(c.Shape()[0] == 3, "MatrixMultiply: Should have correct shape");
    AssertTrue(c.Shape()[1] == 3, "MatrixMultiply: Should have correct shape");
    AssertTrue(c.Get({0, 0}) == 3, "MatrixMultiply: Should compute correct value");
    AssertTrue(c.Get({0, 1}) == 6, "MatrixMultiply: Should compute correct value");
    AssertTrue(c.Get({0, 2}) == 9, "MatrixMultiply: Should compute correct value");
    AssertTrue(c.Get({1, 0}) == 3, "MatrixMultiply: Should compute correct value");
    AssertTrue(c.Get({1, 1}) == 6, "MatrixMultiply: Should compute correct value");
    AssertTrue(c.Get({1, 2}) == 9, "MatrixMultiply: Should compute correct value");
    AssertTrue(c.Get({2, 0}) == 3, "MatrixMultiply: Should compute correct value");
    AssertTrue(c.Get({2, 1}) == 6, "MatrixMultiply: Should compute correct value");
    AssertTrue(c.Get({2, 2}) == 9, "MatrixMultiply: Should compute correct value");
  }
}

void TestTensorZeros() {
  auto t = Zeros<Int32>({3, 3});
  AssertTrue(t.Shape()[0] == 3, "Zeros: Incorrect shape");
  AssertTrue(t.Shape()[1] == 3, "Zeros: Incorrect shape");
  AssertTrue(t.Stride()[0] == 3, "Zeros: Incorrect stride");
  AssertTrue(t.Stride()[1] == 1, "Zeros: Incorrect stride");
  AssertTrue(t.Data().size() == 9, "Zeros: Incorect data size");
  for (auto d : t.Data())
    AssertTrue(d == 0, "Zeros: Elements should all be zero");
}

void TestTensorOnes() {
  auto t = Ones<Int32>({3, 3});
  AssertTrue(t.Shape()[0] == 3, "Ones: Incorrect shape");
  AssertTrue(t.Shape()[1] == 3, "Ones: Incorrect shape");
  AssertTrue(t.Stride()[0] == 3, "Ones: Incorrect stride");
  AssertTrue(t.Stride()[1] == 1, "Ones: Incorrect stride");
  AssertTrue(t.Data().size() == 9, "Ones: Incorect data size");
  for (auto d : t.Data())
    AssertTrue(d == 1, "Ones: Elements should all be one");
}

void TestIdentity() {
  {
    auto t = Identity<Int32>({3, 3});
    AssertTrue(t.At({0, 0}) == 1, "Identity: Diagonal should be 1");
    AssertTrue(t.At({1, 1}) == 1, "Identity: Diagonal should be 1");
    AssertTrue(t.At({2, 2}) == 1, "Identity: Diagonal should be 1");
    AssertTrue(t.At({0, 1}) == 0, "Identity: Off diagonal should be 0");
    AssertTrue(t.At({1, 0}) == 0, "Identity: Off diagonal should be 0");
    AssertTrue(t.At({1, 2}) == 0, "Identity: Off diagonal should be 0");
  }
}

void TestTensorReferenceConstructor() {

}

void TestTensorSlice() {

}

void TestTensorCopy() {

}


int main() {
  TestTensorShapeToStride();
  TestTensorConstructorShapeStride();
  TestTensorSize();
  TestTensorAdd();
  TestTensorMultiply();
  TestTensorSubtract();
  TestTensorNegate();
  TestTensorApply();
  TestTensorGet();
  TestTensorAt();
  TestTensorMatrixMultply();
  TestTensorZeros();
  TestTensorOnes();
  TestIdentity();
  TestTensorSlice();
  TestTensorCopy();
  TestTensorReferenceConstructor();
  return 0;
}