#ifndef JB_TENSOR_H
#define JB_TENSOR_H

#include <vector>
#include <functional>
#include <numeric>
#include <memory>

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

// CONSTRUCTORS
template<typename T>
Tensor<T> Zeros(vector<int> shape) {
  Tensor<T> t;
  t.shape = shape;
  t.stride = ShapeToStrides(shape);
  t.offset = 0;
  t.data = make_shared<vector<T>>(t.Size());
  return t;
}

// TENSOR FRIENDS

template<typename T>
Tensor<T> Add(const Tensor<T> & a, const Tensor<T> & b) {
  Tensor<T> c = Zeros<T>(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data->at(i) = a.data->at(i) + b.data->at(i);
  return c;
}

template<typename T>
Tensor<T> Multiply(const Tensor<T> & a, const Tensor<T> & b) {
  Tensor<T> c = Zeros<T>(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data->at(i) = a.data->at(i) * b.data->at(i);
  return c;
}

template<typename T>
Tensor<T> Subtract(const Tensor<T> & a, const Tensor<T> & b) {
  Tensor<T> c = Zeros<T>(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data->at(i) = a.data->at(i) - b.data->at(i);
  return c;
}

template<typename T>
Tensor<T> Negate(const Tensor<T> & a) {
  Tensor<T> c = Zeros<T>(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data->at(i) = -a.data->at(i);
  return c;
}

template<typename T>
Tensor<T> Apply(const Tensor<T> & a, T (*f)(T)) {
  Tensor<T> c = Zeros<T>(a.shape);
  for (int i = 0; i < c.Size(); i++)
    c.data->at(i) = f(a.data->at(i));
  return c;
}

template<typename T>
Tensor<T> MatrixMultiply(const Tensor<T> & a, const Tensor<T> & b) {
  if (a.NumDimension() != 2)
    throw runtime_error("MatrixMultiply: a is not a matrix");
  if (b.NumDimension() != 2)
    throw runtime_error("MatrixMultiply: b is not a matrix");
  if (a.Shape()[1] != b.Shape()[0])
    throw runtime_error("MatrixMultiply: inner dimensions do not match");

  Tensor<T> c = Zeros<T>({a.Shape()[0], b.Shape()[1]});
  int inner_dim = a.Shape()[1];

  for (int i = 0; i < c.Shape()[0]; i++) {
    for (int j = 0; j < c.Shape()[1]; j++) {
      int val = 0;
      for (int k = 0; k < inner_dim; k++) {
        val += a.Get({i, k}) * b.Get({k, j});
      }
      c.At({i, j}) = val;
    }
  }
  return c;
}

// TENSOR CLASS

template<typename T>
class Tensor {
public:
  Tensor() {};
  friend Tensor Zeros<T>(vector<int> shape);

  const vector<T> & Data() { return (*data); };
  vector<T> & DataMutable() { return (*data); };
  const vector<int> & Shape() const { return shape; };
  const vector<int> & Stride() const { return stride; };
  T Get(vector<int> index) const;
  T & At(vector<int> index);
  int Size();
  int NumDimension() const { return shape.size(); }

  friend Tensor Multiply<T>(const Tensor & a, const Tensor & b);
  friend Tensor Add<T>(const Tensor & a, const Tensor & b);
  friend Tensor Subtract<T>(const Tensor & a, const Tensor & b);
  friend Tensor Negate<T>(const Tensor & a);
  friend Tensor Apply<T>(const Tensor & a, T (*f)(T));
  friend Tensor MatrixMultiply<T>(const Tensor & a, const Tensor & b);

private:
  shared_ptr<vector<T>> data;
  vector<int> shape;
  vector<int> stride;
  int offset;
};

// CONSTRUCTORS

// TENSOR METHODS

template<typename T>
T Tensor<T>::Get(vector<int> index) const {
  int flat_index = 0;
  for (int i = 0; i < index.size(); i++)
    flat_index += stride[i] * index[i];
  return (*data)[flat_index];
}

template<typename T>
T & Tensor<T>::At(vector<int> index) {
  int flat_index = 0;
  for (int i = 0; i < index.size(); i++)
    flat_index += stride[i] * index[i];
  return (*data)[flat_index];
}

template<typename T>
int Tensor<T>::Size() {
  return accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
};

}  // namespace tensor

}  // namespace jb

#endif  // JB_TENSOR_H