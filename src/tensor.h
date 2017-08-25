#ifndef JB_TENSOR_H
#define JB_TENSOR_H

#include <vector>
#include <functional>
#include <numeric>

#define TENSOR_TYPE(type, name) typedef type name;

using namespace std;

namespace jb {

namespace tensor {

TENSOR_TYPE(float, Float32)
TENSOR_TYPE(double, Float64)
TENSOR_TYPE(int8_t, Int8)
TENSOR_TYPE(int16_t, Int16)
TENSOR_TYPE(int32_t, Int32)
TENSOR_TYPE(int64_t, Int64)

template<typename T>
class Tensor;

// UTILITY FUNCTIONS

vector<int> ShapeToStrides(const vector<int> &shape) {
  int ndim = (int)shape.size();
  vector<int> strides(ndim);
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

// FRIEND FUNCTIONS

template<typename T>
Tensor<T> Add(const Tensor<T> & a, const Tensor<T> & b) {
  Tensor<T> c(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data[i] = a.data[i] + b.data[i];
  return c;
}

template<typename T>
Tensor<T> Multiply(const Tensor<T> & a, const Tensor<T> & b) {
  Tensor<T> c(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data[i] = a.data[i] * b.data[i];
  return c;
}

template<typename T>
Tensor<T> Subtract(const Tensor<T> & a, const Tensor<T> & b) {
  Tensor<T> c(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data[i] = a.data[i] - b.data[i];
  return c;
}

template<typename T>
Tensor<T> Negate(const Tensor<T> & a) {
  Tensor<T> c(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data[i] = -a.data[i];
  return c;
}

template<typename T>
Tensor<T> Apply(const Tensor<T> & a, T (*f)(T)) {
  Tensor<T> c(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data[i] = f(a.data[i]);
  return c;
}

template<typename T>
Tensor<T> MatrixMultiply(const Tensor<T> & a, const Tensor<T> & b) {
  // TODO: implement shape functions
}

// TENSOR CLASS
template<typename T>
class Tensor {
public:
  Tensor(vector<int> shape) : shape(shape), stride(ShapeToStrides(shape)) {
    data.resize(Size());
  };

  vector<T> & Data() { return data; };
  vector<int> & Shape() { return shape; };
  vector<int> & Stride() { return stride; };

  int Size() {
    return accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
  };

  friend Tensor Multiply<T>(const Tensor & a, const Tensor & b);
  friend Tensor Add<T>(const Tensor & a, const Tensor & b);
  friend Tensor Subtract<T>(const Tensor & a, const Tensor & b);
  friend Tensor Negate<T>(const Tensor & a);
  friend Tensor Apply<T>(const Tensor & a, T (*f)(T));
  friend Tensor<T> MatrixMultiply(const Tensor<T> & a, const Tensor<T> & b);

private:
  vector<T> data;
  vector<int> shape;
  vector<int> stride;
};

}  // namespace tensor

}  // namespace jb

#endif  // JB_TENSOR_H