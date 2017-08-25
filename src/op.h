#ifndef JB_OP_H
#define JB_OP_H

#include <list>

using namespace std;

namespace jb {

namespace op {

template<typename T>
class Op {
public:
  virtual Tensor<T> jacobian(Op<T> const * const, Op<T> const * const);
private:
  list<Op<T> const * const> inputs;
  list<Op<T> const * const> outputs;
};

}  // namespace op

}  // namespace jb

#endif  // JB_OP_H