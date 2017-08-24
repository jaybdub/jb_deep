#include "src/tensor.h"

namespace jb {

namespace tensor {

// UTILITY FUNCTIONS

vector<int> ShapeToStrides(const vector<int> &shape) {
  int ndim = (int)shape.size();
  vector<int> strides((int)ndim);
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

// FRIEND FUNCTIONS

template<typename T>
Tensor<T> Add(const Tensor<T> & a, const Tensor<T> & b) {
}

template<typename T>
Tensor<T> Multiply(const Tensor<T> & a, const Tensor<T> & b) {
}

// TENSOR CLASS

template<typename T>
Tensor<T>::Tensor(vector<int> shape) : shape(shape), stride(ShapeToStrides(shape)) {
  data.resize(Size());
};

template<typename T>
int Tensor<T>::Size() {
  return accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
}

}  // namespace tensor

}  // namespace jb