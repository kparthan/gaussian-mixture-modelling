#include "Test.h"
#include "Support.h"
#include "MultivariateNormal.h"
#include "SupportUnivariate.h"

void Test::random_data_generation()
{
  MultivariateNormal mvnrm;
  mvnrm.printParameters();

  Vector mean;
  Matrix cov;
  int D;
  int N = 10000;
  std::vector<Vector> random_sample;

  // 2 D
  D = 2;
  mean = Vector(D,0);
  //cov = generateRandomCovarianceMatrix(D);
  cov = IdentityMatrix(2,2); cov(0,0) = 1; cov(1,1) = 10;
  MultivariateNormal mvnorm2d(mean,cov);
  mvnorm2d.printParameters();
  random_sample = mvnorm2d.generate(N);
  writeToFile("./visualize/mvnorm2d.dat",random_sample,3);

  // 3 D
  D = 3;
  mean = Vector(D,0);
  //cov = generateRandomCovarianceMatrix(D);
  cov = IdentityMatrix(3,3); cov(0,0) = 1; cov(1,1) = 10; cov(2,2) = 100;
  MultivariateNormal mvnorm3d(mean,cov);
  mvnorm3d.printParameters();
  random_sample = mvnorm3d.generate(N);
  writeToFile("./visualize/mvnorm3d.dat",random_sample,3);
}

void Test::fisher()
{
  int D;
  Vector mean;
  Matrix cov;

  D = 5;
  mean = Vector(D,0);
  cov = generateRandomCovarianceMatrix(D);
  //cov = IdentityMatrix(3,3); cov(0,0) = 1; cov(1,1) = 10; cov(2,2) = 100;
  MultivariateNormal mvnorm(mean,cov);
  mvnorm.computeLogFisherInformation(1);
}

void Test::determinant()
{
  int D;
  Matrix A,inv;
  long double det;

  D = 5;
  A = generateRandomCovarianceMatrix(D);
  inv = Matrix(D,D);
  invertMatrix(A,inv,det);
  
  cout << "A: " << A << endl;
  cout << "inv(A): " << inv << endl;
  cout << "det(A): " << det << endl;

  A = IdentityMatrix(D);
  inv = Matrix(D,D);
  invertMatrix(A,inv,det);
  
  cout << "A: " << A << endl;
  cout << "inv(A): " << inv << endl;
  cout << "det(A): " << det << endl;
}

void Test::all_estimates_univariate()
{
  Vector random_sample;
  long double mu = 10;
  long double sigma = 2;
  int sample_size = 10;
  struct EstimatesUnivariate estimates;
  Normal norm,norm_est;
  string file_name;

  norm = Normal(mu,sigma);
  norm.printParameters();
  random_sample = norm.generate(sample_size);
  writeToFile("random_sample.dat",random_sample);

  norm_est = Normal(mu,sigma);
  norm_est.computeAllEstimators(random_sample,estimates,1);
}

void Test::all_estimates()
{
  std::vector<Vector> random_sample,means;
  int D;
  Vector mu;
  Matrix cov;
  int sample_size = 100;
  struct Estimates estimates;
  MultivariateNormal mvnorm,mvnorm_est;
  long double sigma = 1;
  string file_name;

  D = 4;
  //means = generateRandomGaussianMeans(1,D);
  //mu = means[0];
  mu = Vector(D,0);
  cov = generateRandomCovarianceMatrix(D);

  mvnorm = MultivariateNormal(mu,cov);
  mvnorm.printParameters();
  random_sample = mvnorm.generate(sample_size);
  writeToFile("random_sample.dat",random_sample,3);

  mvnorm_est = MultivariateNormal(mu,cov);
  mvnorm_est.computeAllEstimators(random_sample,estimates,1);

  /*D = 2;
  mu = Vector(D,0);
  cov = ZeroMatrix(D,D);
  cov(0,0) = 1; cov(1,1) = 10;
  mvnorm = MultivariateNormal(mu,cov);
  mvnorm.printParameters();
  file_name = "./support/R_codes/mvnorm2d.dat";
  random_sample = load_data_table(file_name,D);
  mvnorm_est = MultivariateNormal(mu,cov);
  mvnorm_est.computeAllEstimators(random_sample,estimates,1);

  D = 3;
  mu = Vector(D,0);
  cov = ZeroMatrix(D,D);
  cov(0,0) = 1; cov(1,1) = 10; cov(2,2) = 100;
  mvnorm = MultivariateNormal(mu,cov);
  mvnorm.printParameters();
  file_name = "./support/R_codes/mvnorm3d.dat";
  random_sample = load_data_table(file_name,D);
  mvnorm_est = MultivariateNormal(mu,cov);
  mvnorm_est.computeAllEstimators(random_sample,estimates,1);*/
}

// D = 3
void Test::factor_analysis_spiral_data()
{
  int N = 900;
  std::vector<Vector> data = generate_spiral_data(N);
  Vector dw(N,1);
  Vector mean;
  Matrix cov;
  computeMeanAndCovariance(data,dw,mean,cov);
  cout << "cov: " << cov << endl;

  Vector L;
  Matrix Psi;
  factor_analysis_3d(cov,L,Psi);

  Matrix LL = outer_prod(L,L);
  Matrix rhs = LL + Psi;
  cout << "LL'+Psi: " << rhs << endl;
}

