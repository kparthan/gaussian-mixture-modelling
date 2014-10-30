#include "MultivariateNormal.h"
#include "Support.h"
#include "Normal.h"

MultivariateNormal::MultivariateNormal()
{
  D = 2;
  mean = Vector(2,0);
  cov = IdentityMatrix(2,2);
}

MultivariateNormal::MultivariateNormal(Vector &mean, Matrix &cov) : mean(mean), cov(cov)
{
  D = mean.size();
}

MultivariateNormal MultivariateNormal::operator=(const MultivariateNormal &source)
{
  if (this != &source) {
    D = source.D;
    mean = source.mean;
    cov = source.cov;
  }
  return *this;
}

void MultivariateNormal::printParameters()
{
  cout << "Multivariate Normal Parameters:\n";
  cout << "Mean: "; print(cout,mean,3); cout << endl;
  cout << "Cov: " << cov << endl;
}

std::vector<Vector> MultivariateNormal::generate(int N)
{
  // eigen decomposition of covariance matrix
  Vector eigen_values(D,0);
  Matrix eigen_vectors = IdentityMatrix(D,D);
  eigenDecomposition(cov,eigen_values,eigen_vectors);
  Matrix sqrt_diag = IdentityMatrix(D,D);
  for (int i=0; i<D; i++) {
    sqrt_diag(i,i) = sqrt(eigen_values[i]);
  }
  Matrix A = prod(eigen_vectors,sqrt_diag);

  Normal normal;
  Vector x(D,0);
  std::vector<Vector> random_sample;
  for (int i=0; i<N; i++) {
    Vector z = normal.generate(D);
    // x = mu + A z
    Vector az = prod(A,z);
    for (int j=0; j<D; j++) {
      x[j] = mean[j] + az[j];
    }
    random_sample.push_back(x);
  }
  return random_sample;
}

