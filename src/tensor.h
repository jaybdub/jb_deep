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

template<typename T>
Tensor<T> Ones(vector<int> shape) {
  Tensor<T> t;
  t.shape = shape;
  t.stride = ShapeToStrides(shape);
  t.offset = 0;
  t.data = make_shared<vector<T>>(t.Size());
  for (auto &d : t.DataMutable())
    d = 1;
  return t;
}

template<typename T>
Tensor<T> Identity(vector<int> shape) {
  Tensor<T> t = Zeros<T>(shape);
  int min_dim = shape[0];
  for (int i = 1; i < shape.size(); i++) {
    if (shape[i] < min_dim)
      min_dim = shape[i];
  }
  for (int i = 0; i < min_dim; i++) {
    vector<int> index(shape.size());
    for (auto &it : index)
      it = i;
    t.At(index) = 1;
  }
  return t;
}

template<typename T>
Tensor<T> RandomNormal(vector<int> shape, T mean, T stdev) {

}

template<typename T>
Tensor<T> RandomUniform(vector<int> shape, T min, T max) {

}

template<typename T>
Tensor<T> Slice(const Tensor<T> & other, vector<int> start, vector<int>
    stop, vector<int> stride) {
  // offset = other offset + start
  // stride = other stride * stride
  // shape = ?
}

template<typename T>
Tensor<T> Copy(const Tensor<T> & src) {
  Tensor<T> dst = Zeros<T>(src.shape);
  MoveHelper(src, dst, src.offset, dst.offset, 0);
  return dst;
}

template<typename T>
void Move(const Tensor<T> & src, Tensor<T> & dst) {
  MoveHelper(src, dst, src.offset, dst.offset, 0);
}

template<typename T>
void MoveHelper(const Tensor<T> & a, Tensor<T> & b, int da, int db, int dim) {
  if (dim < a.NumDimension()) {
    int stride_a = a.stride[dim];
    int stride_b = b.stride[dim];
    int shape = a.shape[dim];
    for (int i = 0; i < shape; i++) {
      MoveHelper<T>(a, b, da + i * stride_a, db + i * stride_b, dim + 1);
    }
  } else {
    b.data->at(db) = a.data->at(da);
  }
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
  // Constructors
  Tensor() {};
  Tensor(const Tensor<T> & other) {
    data = other.data;
    stride = other.stride;
    shape = other.shape;
    offset = other.offset;
  };

  friend Tensor Zeros<T>(vector<int> shape);
  friend Tensor Ones<T>(vector<int> shape);
  friend Tensor Identity<T>(vector<int> shape);
  friend Tensor RandomNormal<T>(vector<int> shape, T mean, T stdev);
  friend Tensor RandomUniform<T>(vector<int> shape, T min, T max);
  friend Tensor Slice<T>(const Tensor<T> & other, vector<int> start, vector<int>
    stop, vector<int> stride);
  friend Tensor Copy<T>(const Tensor<T> & other);
  friend void Move<T>(const Tensor<T> & src, Tensor<T> & dst);
  friend void MoveHelper<T>(const Tensor<T> & a, Tensor<T> & b, int da, int db,
                            int dim);

  // Getters
  const vector<T> & Data() { return (*data); };
  vector<T> & DataMutable() { return (*data); };
  const vector<int> & Shape() const { return shape; };
  const vector<int> & Stride() const { return stride; };
  T Get(vector<int> index) const;
  T & At(vector<int> index);
  int Size();
  int NumDimension() const { return shape.size(); }
  int DataIndex(vector<int> index) const;

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
int Tensor<T>::DataIndex(vector<int> index) const {
  int flat_index = offset;
  for (int i = 0; i < index.size(); i++)
    flat_index += stride[i] * index[i];
  return flat_index;
}

template<typename T>
T Tensor<T>::Get(vector<int> index) const {
  return (*data)[DataIndex(index)];
}

template<typename T>
T & Tensor<T>::At(vector<int> index) {
  return (*data)[DataIndex(index)];
}

template<typename T>
int Tensor<T>::Size() {
  return accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
};

}  // namespace tensor

}  // namespace jb

#endif  // JB_TENSOR_H