#ifndef MULTIVARIATE_NORMAL_H
#define MULTIVARIATE_NORMAL_H

#include "Header.h"

class MultivariateNormal
{
  private:
    int D;

    Vector mean;

    Matrix cov; // +ve def symmetric matrix

  public:
    MultivariateNormal();

    MultivariateNormal(Vector &, Matrix &);

    MultivariateNormal operator=(const MultivariateNormal &);

    void printParameters();

    std::vector<Vector> generate(int);
};

#endif

