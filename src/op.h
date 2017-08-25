#ifndef JB_OP_H
#define JB_OP_H

#include <list>
#include <map>

#include "src/tensor.h"
#include "src/session.h"

using namespace std;
using namespace jb::tensor;

namespace jb {

namespace op {

template<typename T>
class Session;

template<typename T>
class Op {
public:
  Op() {};
  Op(list<Op<T> *> inputs) : inputs(inputs) {};
  virtual const Tensor<T> & Evaluate(Session<T> & session) = 0;
private:
  list<Op<T> *> inputs;
};

// OP SUBCLASSES

template<typename T>
class Variable : public Op<T> {
public:
  const Tensor<T> & Evaluate(Session<T> & session) override {
    return session.values[this];
  }
  const Tensor<T> & Assign(Session<T> & session, Tensor<T> value) {
    session.values[this] = value;
    return session.values[this];
  }
};

template<typename T>
class Add : public Op<T> {
  const Tensor<T> & Evaluate(Session<T> & session) override {
    // evaluate inputs
    for (auto input : inputs) {
      input->Evaluate(session);
    }
    return session.values[this];
  }
};

}  // namespace op

}  // namespace jb

#endif  // JB_OP_H