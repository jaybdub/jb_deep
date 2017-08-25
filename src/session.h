#ifndef JB_SESSION_H
#define JB_SESSION_H

#include <list>
#include <unordered_map>
#include "src/op.h"
#include "src/tensor.h"

using namespace std;
using namespace jb::op;
using namespace jb::tensor;

namespace jb {

namespace session {

// Running operations and cache results.  Only necessary computations are
// performed.

template<typename T>
class Session {
public:
  void Run(list<Op<T> *> outputs);
  void Assign(Variable<T> *, Tensor<T>);
  const unordered_map<Op<T> *, Tensor<T>> & Values() { return values; };
private:
  void Evaluate(Op<T> *, long run);
  unordered_map<Op<T> *, Tensor<T>> values;
  unordered_map<Op<T> *, long> runs;
  long run = 0;
};

template<typename T>
void Session<T>::Assign(Variable<T> * variable, Tensor<T> value) {
  variable->Assign(values, value);
}

template<typename T>
void Session<T>::Run(list<Op<T> *> outputs) {
  run++;
  for (auto o : outputs)
    Evaluate(o, run);
}

template<typename T>
void Session<T>::Evaluate(Op<T> * op, long run) {
  if (runs[op] == run) {
    return;  // result cached for this run
  } else {
    // run dependents
    for (auto input : op->Inputs()) {
      Evaluate(input, run);
    }
    // evaluate op (assumes dependents exist)
    op->Evaluate(values);
  }
}

}  // namespace session

}  // namespace jb

#endif  // JB_SESSION_H