#ifndef JB_OP_H
#define JB_OP_H

#include <list>
#include <unordered_map>

#include "src/tensor.h"

using namespace std;
using namespace jb::tensor;

namespace jb {

namespace op {

template<typename T>
class Op {
public:
  Op() {};
  Op(list<Op<T> *> inputs) : inputs(inputs) {};
  virtual const Tensor<T> & Evaluate(unordered_map<Op<T> *, Tensor<T>> &) = 0;
  const list<Op<T> *> & Inputs() { return inputs; };
private:
  list<Op<T> *> inputs;
};

// OP SUBCLASSES

template<typename T>
class Variable : public Op<T> {
public:
  const Tensor<T> & Evaluate(unordered_map<Op<T> *, Tensor<T>> & values) {
    // assumes values computed.
    return values[this];
  }
  const Tensor<T> & Assign(unordered_map<Op<T> *, Tensor<T>> & values,
                           Tensor<T> value) {
    values[this] = value;
    return values[this];
  }
};

}  // namespace op

}  // namespace jb

#endif  // JB_OP_H