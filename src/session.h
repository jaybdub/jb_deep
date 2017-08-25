#ifndef JB_SESSION_H
#define JB_SESSION_H

namespace jb {
namespace op {
template<typename T>
class Op;
}
}
#include <list>
#include <unordered_map>
#include "src/op.h"
#include "src/tensor.h"

using namespace std;
using namespace jb::op;
using namespace jb::tensor;

namespace jb {

namespace op {

// Running operations and cache results.  Only necessary computations are
// performed.
template<typename T>
class Op;
template<typename T>
class Variable;

template<typename T>
class Session {
public:
  friend class Op<T>;
  friend class Variable<T>;
private:
  unordered_map<Op<T> *, Tensor<T>> values;
};

}  // namespace session

}  // namespace jb

#endif  // JB_SESSION_H