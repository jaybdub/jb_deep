#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace std;

struct TensorShape {
  TensorShape() {};
  TensorShape(vector<int> shape) : shape(shape) {};
  vector<int> shape;

  int Rank() const {
    return shape.size();
  }

  int NumElements() {
    int size = 1;
    for (auto dim : shape)
      size *= dim;
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
  inline int Dim(int dim) {
    return indices[dim];
  }
};

template<typename T>
struct Tensor {
  Tensor() {};
  Tensor(TensorShape shape) : shape(shape) {
    data.resize(this->shape.NumElements());
    strides = ShapeToStrides(shape);
  };
  Tensor(vector<int> shape) : shape(shape) {
    data.resize(this->shape.NumElements());
    strides = ShapeToStrides(shape);
  }
  vector<T> data;
  TensorShape shape;
  vector<int> strides;

  void print() {
    cout << "\nNumElements: ";
    cout << shape.NumElements();
    cout << "\nShape: ";
    for (auto s : shape.shape)
      cout << s << ' ';
    cout << "\nStrides: ";
    for (auto s : strides)
      cout << s << ' ';
    cout << "\nData: ";
    for (auto d : data)
      cout << d << ' ';
    cout << endl;
  }

  inline T & at(TensorIndex index) {
    int flat_index = 0;
    for (int i = 0; i < shape.Rank(); i++)
      flat_index += strides[i] * index.Dim(i);
    return data[flat_index];
  }

  static vector<int> ShapeToStrides(TensorShape shape) {
    vector<int> strides(shape.Rank());
    strides[shape.Rank() - 1] = 1;
    for (int i = shape.Rank() - 2; i >= 0; i--) {
      strides[i] = shape.shape[i + 1] * strides[i + 1];
    }
    return strides;
  }

  static Tensor<T> Add(const Tensor<T> & a, const Tensor<T> & b) {
    Tensor<T> out(a.shape);
    for (int i = 0; i < out.shape.NumElements(); i++)
      out.data[i] = a.data[i] + b.data[i];
    return out;
  }

  static Tensor<T> Mult(const Tensor<T> & a, const Tensor<T> & b) {
    Tensor<T> out(a.shape);
    for (int i = 0; i < out.shape.NumElements(); i++)
      out.data[i] = a.data[i] * b.data[i];
    return out;
  }

  static Tensor<T> MatMult(Tensor<T> & a, Tensor<T> & b) {
    if (a.shape.shape[1] != b.shape.shape[0])
      throw runtime_error("Invalid dimensions");
    TensorShape tshape({a.shape.shape[0], b.shape.shape[1]});
    Tensor<T> out(tshape);
    int ni = out.shape.shape[0];
    int nj = out.shape.shape[1];
    int nk = out.shape.shape[2];
    for (int i = 0; i < ni; i++) {
      for (int j = 0; j < nj; j++) {
        int sum = 0;
        for (int k = 0; k < nk; k++) {
          sum += a.at(TensorIndex({i, k})) * b.at(TensorIndex({k, j}));
        }
        out.at(TensorIndex({i, j})) = sum;
      }
    }
    return out;
  }

  static Tensor<T> Ones(TensorShape shape) {
    Tensor<T> t(shape);
    for (int i = 0; i < t.shape.NumElements(); i++)
      t.data[i] = 1;
    return t;
  }

  static Tensor<T> Zeros(TensorShape shape) {
    Tensor<T> t(shape);
    for (int i = 0; i < t.shape.NumElements(); i++)
      t.data[i] = 0;
    return t;
  }
};

template<typename T>
struct Op {
  // evaluates the output of the operation by recursively evaluating the output
  // of dependent operations until a Variable operation is reached.
  virtual Tensor<T> evaluate() = 0;

  // evaluates the partial derivative of the operation wrt. another operation.
  //
  // (output shape, input shape)
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
struct MatMult : public Op<T> {
  MatMult(Op<T>* a, Op<T>* b) : a(a), b(b) {};
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
  Tensor<float> t1({3, 3});
  Tensor<float> t2({3, 1});
  t1.data = {1, 2, 3, 1, 2, 3, 1, 2, 3};
  t2.data = {1, 1, 1};
  Tensor<float> t3 = Tensor<float>::MatMult(t1, t2);
  t1.print();
  t2.print();
  t3.print();
  return 0;
}