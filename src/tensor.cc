#include "src/tensor.h"

namespace jb {

vector<int> Tensor::ShapeToStrides(const vector<int> &shape) {
  int ndim = shape.size();
  vector<int> strides(ndim);
  strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >=0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  return strides;
}

}  // namespace jb