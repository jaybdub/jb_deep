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

// TENSOR CLASS

}  // namespace tensor

}  // namespace jb