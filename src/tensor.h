#ifndef JB_TENSOR_H
#define JB_TENSOR_H

#include <vector>
#include <functional>
#include <numeric>

#define TENSOR_TYPE(type, name) typedef type name; template class Tensor<name>;

using namespace std;

namespace jb {

namespace tensor {

template<typename T>
class Tensor;

// UTILITY FUNCTIONS

vector<int> ShapeToStrides(const vector<int> & shape);

// FRIEND FUNCTIONS

template<typename T>
Tensor<T> Add(const Tensor<T> & a, const Tensor<T> & b);

template<typename T>
Tensor<T> Multiply(const Tensor<T> & a, const Tensor<T> & b);

// TENSOR CLASS
template<typename T>
class Tensor {
public:
  Tensor(vector<int> shape);
  vector<T> & Data() { return data; };
  vector<int> & Shape() { return shape; };
  vector<int> & Stride() { return stride; };
  int Size();
  friend Tensor Multiply<T>(const Tensor & a, const Tensor & b);
  friend Tensor Add<T>(const Tensor & a, const Tensor & b);

private:
  vector<T> data;
  vector<int> shape;
  vector<int> stride;
};

// DATA TYPES
TENSOR_TYPE(float, Float32);
TENSOR_TYPE(double, Float64);
TENSOR_TYPE(int8_t, Int8);
TENSOR_TYPE(int16_t, Int16);
TENSOR_TYPE(int32_t, Int32);
TENSOR_TYPE(int64_t, Int64);
TENSOR_TYPE(uint8_t, UInt8);
TENSOR_TYPE(uint16_t, UInt16);
TENSOR_TYPE(uint32_t, UInt32);
TENSOR_TYPE(uint64_t, UInt64);

}  // namespace tensor

}  // namespace jb

#endif  // JB_TENSOR_H