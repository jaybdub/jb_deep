#include <iostream>
#include "src/tensor.h"
#include "src/op.h"
#include "test/test.h"
#include "src/session.h"

using namespace std;
using namespace jb;
using namespace jb::tensor;
using namespace jb::op;
using namespace jb::test;
using namespace jb::session;

void TestVariable() {
  {
    Variable<Int32> a;
    Tensor<Int32> a_val({3});
    a_val.DataMutable() = {1, 2, 3};
    unordered_map<Op<Int32> *, Tensor<Int32>> values;
    a.Assign(values, a_val);
    auto a_res = a.Evaluate(values);
    AssertTrue(a_res.DataMutable()[0] == 1, "Invalid assign data");
    AssertTrue(a_res.DataMutable()[1] == 2, "Invalid assign data");
    AssertTrue(a_res.DataMutable()[2] == 3, "Invalid assign data");
  }
}

void TestSessionRun() {
  {
    // create session
    Session<Int32> s;

    // create variable and assign value in session
    Variable<Int32> a;
    Tensor<Int32> a_val({3});
    a_val.DataMutable() = {1, 2, 3};
    s.Assign(&a, a_val);

    // run session
    s.Run({&a});

    // get result
    auto values = s.Values();
    AssertTrue(values[&a].Data()[0] == 1, "Should store values in session");
    AssertTrue(values[&a].Data()[1] == 2, "Should store values in session");
    AssertTrue(values[&a].Data()[2] == 3, "Should store values in session");
  }
}

int main() {
  TestVariable();
  TestSessionRun();
  return 0;
}