#include <iostream>
#include <list>
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

void TestAdd() {
  {
    Session<Int32> s;
    Variable<Int32> a, b;
    op::Add<Int32> add({&a, &b});
    Tensor<Int32> val({3, 2});
    val.DataMutable() = {1, 2, 3, 4, 5, 6};
    s.Assign(&a, val);
    s.Assign(&b, val);
    s.Run({&add});
    auto values = s.Values();
    AssertTrue(values[&add].DataMutable()[0] == 2, "Invalid add value");
    AssertTrue(values[&add].DataMutable()[1] == 4, "Invalid add value");
    AssertTrue(values[&add].DataMutable()[2] == 6, "Invalid add value");
    AssertTrue(values[&add].DataMutable()[3] == 8, "Invalid add value");
    AssertTrue(values[&add].DataMutable()[4] == 10, "Invalid add value");
    AssertTrue(values[&add].DataMutable()[5] == 12, "Invalid add value");
  }
}

void TestMultiply() {
  {
    Session<Int32> s;
    Variable<Int32> a, b;
    op::Multiply<Int32> multiply({&a, &b});
    Tensor<Int32> val({3, 2});
    val.DataMutable() = {1, 2, 3, 4, 5, 6};
    s.Assign(&a, val);
    s.Assign(&b, val);
    s.Run({&multiply});
    auto values = s.Values();
    AssertTrue(values[&multiply].DataMutable()[0] == 1, "Invalid multiply value");
    AssertTrue(values[&multiply].DataMutable()[1] == 4, "Invalid multiply value");
    AssertTrue(values[&multiply].DataMutable()[2] == 9, "Invalid multiply value");
    AssertTrue(values[&multiply].DataMutable()[3] == 16, "Invalid multiply value");
    AssertTrue(values[&multiply].DataMutable()[4] == 25, "Invalid multiply value");
    AssertTrue(values[&multiply].DataMutable()[5] == 36, "Invalid multiply value");
  }
}


int main() {
  TestVariable();
  TestSessionRun();
  TestAdd();
  TestMultiply();
  return 0;
}