#ifndef NORMAL_H
#define NORMAL_H

#include "Header.h"

class Normal
{
  private:
    //! Mean of the distribution
		long double mu;

    //! Standard deviation of the distribution
		long double sigma;

    long double log_cd;

  public:
		//! Constructor
		Normal() ;

		//! Constructor that sets value of parameters
		Normal(long double, long double);

    void updateConstants();

    long double computeLogNormalizationConstant();

    //! Assignment of an existing Normal distribution
    Normal operator=(const Normal &);

    void printParameters();

    void printParameters(ostream &);

    void printParameters(ostream &, int);

    //! Generate random sample
    Vector generate(int);

    void computeAllEstimators(Vector &, struct EstimatesUnivariate &, int verbose = 0);

    void estimateParameters(Vector &, Vector &);

    void estimateMean(struct EstimatesUnivariate &, Vector &, Vector &);

    void estimateSigma(struct EstimatesUnivariate &, Vector &, Vector &);

    void updateParameters(struct EstimatesUnivariate &);

		//! Gets the mean 
		long double Mean();

    //! Gets the standard deviation
    long double Sigma(); 

    long double getLogNormalizationConstant();

		//! Function value
		long double log_density(long double &);

    //! Computes the negative log likelihood of a sample
    long double computeNegativeLogLikelihood(long double &);

    //! Computes the negative log likelihood of a sample
    long double computeNegativeLogLikelihood(Vector &);

    long double computeMessageLength(Vector &);

    long double computeLogParametersProbability(long double);

    long double computeLogPriorDensity();

    long double computeLogFisherInformation(long double);

    long double entropy();

    long double computeKLDivergence(Normal &);

    Normal conflate(Normal &);
};

#endif

