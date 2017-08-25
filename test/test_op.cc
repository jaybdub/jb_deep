#include <iostream>
#include "src/tensor.h"
#include "src/op.h"
#include "src/session.h"
#include "test/test.h"

using namespace std;
using namespace jb;
using namespace jb::tensor;
using namespace jb::op;
using namespace jb::test;

void TestVariable() {
  {
    Session<Int32> s;
    Variable<Int32> v;
    Tensor<Int32> v_val({3});
    v_val.DataMutable() = {1, 2, 3};
    v.Assign(s, v_val);
    auto result = v.Evaluate(s);
    AssertTrue(result.DataMutable()[0] == 1, "Variable value should match");
    AssertTrue(result.DataMutable()[1] == 2, "Variable value should match");
    AssertTrue(result.DataMutable()[2] == 3, "Variable value should match");
  }
}

int main() {
  TestVariable();
  return 0;
}