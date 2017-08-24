#ifndef JB_TEST_H
#define JB_TEST_H

using namespace std;

namespace jb {

namespace test {

void AssertTrue(bool condition, string message) {
  if (!condition)
    throw runtime_error(message);
}

}  // namespace test

}  // namespace jb

#endif  // JB_TEST_H