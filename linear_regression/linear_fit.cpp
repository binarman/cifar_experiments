#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <sstream>
#include <libalglib/solvers.h>

using namespace std;

struct Data{
// note index 0 should be always 1
  vector<float> x;
  float y;
};

struct Equation{
  Equation(int n): c(n), A(n){
    for (int i = 0; i < n; ++i)
      A[i].resize(n);
  }

  int size() const{
    return c.size();
  }
  vector<float> c;
  vector<vector<float>> A;
};

using Plane = vector<float>;

Plane generatePlane(int n){
  default_random_engine eng;
  uniform_real_distribution<float> dist(-10, 10);
  Plane p(n);
  for (int i=0; i<n; ++i)
    p[i] = dist(eng);
  return p;
}

float evaluatePlane(const Plane &p, const vector<float> &args){
  float result = inner_product(p.begin(), p.end(), args.begin(), 0.0);
  return result;
}

string toString(const vector<float> &v) {
  stringstream os;
  os << "[";
  for (int i=0; i<v.size(); ++i) {
    if (i!=0)
      os << ", ";
    os << v[i];
  }
  os << "]";
  return os.str();
}

vector<Data> generateDataset(const Plane &p, int dataset_size) {
  default_random_engine eng;
  uniform_real_distribution<float> dist(-10, 10);
  size_t plane_size = p.size();
  vector<Data> dataset;
  for (int i=0; i<dataset_size; ++i){
    Data item;
    item.x.resize(plane_size);
    item.x[0] = 1;
    for (int j=1; j<plane_size; ++j){
      item.x[j] = dist(eng);
    }
    item.y = evaluatePlane(p, item.x);
    dataset.push_back(item);
  }
  return dataset;
}

Equation createEquationFromData(const vector<Data> dataset){
  size_t n = dataset[0].x.size();
  size_t dataset_size = dataset.size();
  Equation e(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float Aij = 0;
      for (int k = 0; k < dataset_size; ++k)
        Aij += dataset[k].x[i]*dataset[k].x[j];
      e.A[i][j] = Aij;
    }
    float cst = 0;
    for (int k = 0; k < dataset_size; ++k)
      cst += dataset[k].x[i]*dataset[k].y;
    e.c[i] = cst;
  }
  return e;
}

vector<float> getVectorError(const Equation &e, Plane &p){
  int n = e.size();
  vector<float> error(n);
  for(int i = 0; i < n; ++i)
    error[i] = inner_product(e.A[i].begin(), e.A[i].end(), p.begin(), 0.0) - e.c[i];
  return error;
}

float getSquareError(const Equation &e, Plane &p){
  vector<float> errors = getVectorError(e, p);
  return inner_product(errors.begin(), errors.end(), errors.begin(), 0.0f);
}

Plane solveEquation(const Equation &e) {
  alglib::real_2d_array A;
  size_t n = e.size();
  A.setlength(n, n);
  for (int i=0; i<n; ++i)
    for (int j=0; j<n; ++j)
      A(i,j) = e.A[i][j];
  alglib::real_1d_array c;
  c.setlength(n);
  for (int i=0; i<n; ++i)
    c(i) = e.c[i];

  alglib::ae_int_t info;
  alglib::densesolverreport rep;
  alglib::real_1d_array x;
  x.setlength(n);

  alglib::rmatrixsolve(A, n, c, info, rep, x);
  assert(info > 0);


  Plane p(n);
  for (int i=0; i<n; ++i)
    p[i] = x[i];
  return p;
}

int main(){
  Plane p = generatePlane(10);
  cout << "original plane: " << toString(p) << "\n";
  vector<Data> dataset = generateDataset(p, 100);

  Equation e = createEquationFromData(dataset);
  vector<float> vector_error = getVectorError(e, p);
  cout << "vector or errors: " << toString(vector_error) << "\n";

  float square_error = getSquareError(e, p);
  cout << "square error: " << square_error << "\n";

  Plane solution = solveEquation(e); 
  cout << "solution: " << toString(solution) << "\n";
  return 0;
}

