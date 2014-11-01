#ifndef MULTIVARIATE_NORMAL_H
#define MULTIVARIATE_NORMAL_H

#include "Header.h"

// pdf = cd * exp( ... )
class MultivariateNormal
{
  private:
    int D;

    Vector mu;

    Matrix cov,cov_inv;

    long double det_cov,log_cd;

  public:
    MultivariateNormal();

    MultivariateNormal(Vector &, Matrix &);

    void updateConstants();

    long double computeLogNormalizationConstant();

    MultivariateNormal operator=(const MultivariateNormal &);

    void printParameters();

    void printParameters(ostream &);

    std::vector<Vector> generate(int);

    void computeAllEstimators(std::vector<Vector> &, struct Estimates &, int verbose = 0);

    void estimateParameters(std::vector<Vector> &, Vector &);

    void estimateMean(struct Estimates &, std::vector<Vector> &, Vector &);

    void estimateCovariance(struct Estimates &, std::vector<Vector> &, Vector &);

    void updateParameters(struct Estimates &);

    Vector Mean();

    Matrix Covariance();

    Matrix CovarianceInverse();

    long double getLogNormalizationConstant();

    long double log_density(Vector &);

    long double computeNegativeLogLikelihood(Vector &);

    long double computeNegativeLogLikelihood(std::vector<Vector> &);

    long double computeMessageLength(std::vector<Vector> &);

    long double computeLogParametersProbability(long double);

    long double computeLogPriorDensity();

    long double computeLogFisherInformation(long double);

    long double computeKLDivergence(MultivariateNormal &);
};

#endif

