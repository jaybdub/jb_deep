#ifndef JB_TENSOR_H
#define JB_TENSOR_H

#include <vector>

using namespace std;

namespace jb {

template<typename T>
struct Tensor {

  Tensor(vector<int> shape) : shape(shape) {
    strides = ShapeToStrides(shape);
  }

  vector<T> data;
  vector<int> shape;
  vector<int> strides;

  static vector<int> ShapeToStrides(const vector<int> & shape);

};

}  // namespace jb

#endif  // JB_TENSOR_H