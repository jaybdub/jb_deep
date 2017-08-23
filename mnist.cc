#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace std;

struct TensorShape {
  TensorShape() {};
  TensorShape(vector<int> shape) : shape(shape) {};
  vector<int> shape;
  int GetSize() {
    int size = 0;
    for (auto dim : shape)
      size += dim;
    return size;
  }
  static TensorShape Slice(const TensorShape& a, int start, int end) {
    vector<int> slice_v;
    for (int i = start; i < end; i++) {
      slice_v.push_back(a.shape[i]);
    }
    return TensorShape(slice_v);
  }

  static TensorShape Concatenate(const TensorShape& a, const TensorShape& b) {
    vector<int> concat;
    for (auto ele : a.shape)
      concat.push_back(ele);
    for (auto ele : b.shape)
      concat.push_back(ele);
    return TensorShape(concat);
  }
};

struct TensorIndex {
  TensorIndex(vector<int> indices) : indices(indices) {};
  vector<int> indices;
};

template<typename T>
struct Tensor {
  Tensor() {};
  Tensor(TensorShape shape) : shape(shape) {
    data.resize(this->shape.GetSize());
  };
  Tensor(vector<int> shape) : shape(shape) {
    data.resize(this->shape.GetSize());
  }
  vector<T> data;
  TensorShape shape;

  void print() {
    for (auto d : data)
      cout << d << ' ';
    cout << endl;
  }

  static Tensor<T> Add(const Tensor<T> & a, const Tensor<T> & b) {
    Tensor<T> out(a.shape);
    for (int i = 0; i < out.shape.GetSize(); i++)
      out.data[i] = a.data[i] + b.data[i];
    return out;
  }

  static Tensor<T> Mult(const Tensor<T> & a, const Tensor<T> & b) {
    Tensor<T> out(a.shape);
    for (int i = 0; i < out.shape.GetSize(); i++)
      out.data[i] = a.data[i] * b.data[i];
    return out;
  }

  static Tensor<T> Ones(TensorShape shape) {
    Tensor<T> t(shape);
    for (int i = 0; i < t.shape.GetSize(); i++)
      t.data[i] = 1;
    return t;
  }

  static Tensor<T> Zeros(TensorShape shape) {
    Tensor<T> t(shape);
    for (int i = 0; i < t.shape.GetSize(); i++)
      t.data[i] = 0;
    return t;
  }
};

template<typename T>
struct Op {
  virtual Tensor<T> evaluate() = 0;
  virtual Tensor<T> partial(Op<T> * x) = 0;
};

template<typename T>
struct Variable : public Op<T> {
  Variable(vector<int> shape) : tensor(shape) {};
  Tensor<T> tensor;
  Tensor<T> evaluate() override {
    return tensor;
  }
  Tensor<T> partial(Op<T> * x) {
    return Tensor<T>::Zeros(x->evaluate().shape);
  }
};

template<typename T>
struct Dot : public Op<T> {
  Dot(Op<T>* a, Op<T>* b) : a(a), b(b) {};
  Op<T>* a;
  Op<T>* b;
  Tensor<T> evaluate() override {
    // run dependent outputs
    Tensor<T> a_out = a->evaluate();
    Tensor<T> b_out = b->evaluate();

    // allocate output tensor
    TensorShape out_shape = TensorShape::Concatenate(
        TensorShape::Slice(a_out.shape, 0, (int)a_out.shape.shape.size() - 1),
        TensorShape::Slice(b_out.shape, 1, (int)b_out.shape.shape.size())
    );
    Tensor<T> out(out_shape);
    // perform dot product
  }
};

template<typename T>
struct Add : public Op<T> {
  Add(Op<T> * a, Op<T> * b) : a(a), b(b) {};
  Op<T> * a, * b;

  Tensor<T> evaluate() override {
    return Tensor<T>::Add(a->evaluate(), b->evaluate());
  }

  Tensor<T> partial(Op<T> * x) override {
    if (x == a) {
      // dy/dx = dy/da
      return Tensor<T>::Ones(a->evaluate().shape);
    } else if (x == b) {
      // dy/dx = dy/db
      return Tensor<T>::Ones(b->evaluate().shape);
    } else {
      // dy/dx = dy/da * da/dx + dy/db * db/dx
      //       = da/dx + db/dx
      //       = a->partial(x) + b->partial(x)
      return Tensor<T>::Add(a->partial(x), b->partial(x));
    }
  }
};

template<typename T>
struct Mult : public Op<T> {
  Mult(Op<T> * a, Op<T> * b) : a(a), b(b) {};
  Op<T> * a, * b;

  Tensor<T> evaluate() {
    // y = a * b
    return Tensor<T>::Mult(a->evaluate(), b->evaluate());
  }

  Tensor<T> partial(Op<T> * x) {
    // y = a * b
    // if x = a
    //   dy/dx = dy/da = b
    // if x = b
    //   dy/dx = dy/db = a
    // otherwise
    //   dy/dx = dy/da * da/dx + dy/db * db/dx
    //         = b * da/dx + a * db/dx
    //         = b * a->partial(x) + a * b->partial(x)
    if (x == a) {
      return b->evaluate();
    } else if (x == b) {
      return a->evaluate();
    } else {
      return Tensor<T>::Add(
          Tensor<T>::Mult(b->evaluate(), a->partial(x)),
          Tensor<T>::Mult(a->evaluate(), b->partial(x))
      );
    }
  }
};

int main() {
  Variable<float> a({3});
  Variable<float> b({3});
  a.tensor.data = {1, 2, 3};
  b.tensor.data = {4, 5, 6};
  Add<float> add(&a, &b);
  Mult<float> mult(&a, &b);
  auto add_out = add.evaluate();
  auto mult_out = mult.evaluate();

  add_out.print();
  mult_out.print();

  auto dadd_da = add.partial(&a);
  auto dadd_db = add.partial(&b);
  auto dmult_da = mult.partial(&a);
  auto dmult_db = mult.partial(&b);
  dadd_da.print();
  dadd_db.print();
  dmult_da.print();
  dmult_db.print();


  return 0;
}