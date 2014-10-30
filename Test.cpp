#include "Test.h"
#include "Support.h"
#include "MultivariateNormal.h"

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

