#ifndef JB_OP_H
#define JB_OP_H

#include <vector>
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
  virtual const Tensor<T> & Evaluate(unordered_map<Op<T> *, Tensor<T>> &) = 0;
  virtual vector<Op<T> *> Inputs() = 0;
};

// OP SUBCLASSES

template<typename T>
class Variable : public Op<T> {
public:
  const Tensor<T> & Evaluate(unordered_map<Op<T> *, Tensor<T>> & values) override {
    // assumes values computed.
    return values[this];
  }
  const Tensor<T> & Assign(unordered_map<Op<T> *, Tensor<T>> & values,
                           Tensor<T> value) {
    values[this] = value;
    return values[this];
  }
  vector<Op<T> *> Inputs() { return {}; }
};

template<typename T>
class Add : public Op<T> {
public:
  Add(vector<Op<T> *> inputs) : inputs(inputs) {};
  const Tensor<T> & Evaluate(unordered_map<Op<T> *, Tensor<T>> & values) override {
    values[this] = values[Inputs()[0]];
    for (int i = 1; i < Inputs().size(); i++)
      values[this] = tensor::Add(values[this], values[Inputs()[i]]);
    return values[this];
  }
  vector<Op<T> *> Inputs() { return inputs; };
private:
  vector<Op<T> *> inputs;
};

template<typename T>
class Multiply : public Op<T> {
public:
  Multiply(vector<Op<T> *> inputs) : inputs(inputs) {};
  const Tensor<T> & Evaluate(unordered_map<Op<T> *, Tensor<T>> & values) override {
    values[this] = values[Inputs()[0]];
    for (int i = 1; i < Inputs().size(); i++)
      values[this] = tensor::Multiply(values[this], values[Inputs()[i]]);
    return values[this];
  }
  vector<Op<T> *> Inputs() { return inputs; };
private:
  vector<Op<T> *> inputs;
};

}  // namespace op

}  // namespace jb

#endif  // JB_OP_H