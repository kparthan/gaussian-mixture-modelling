#ifndef SUPPORT_UNIVARIATE_H
#define SUPPORT_UNIVARIATE_H

#include "Support.h"
#include "Normal.h"
#include "MixtureUnivariate.h"

struct EstimatesUnivariate {
  long double Neff;
  long double mean;
  long double sigma_ml,sigma_mml;
};

long double computeSigma(Vector &, Vector &, long double &);
void computeMeanAndSigma(Vector &, Vector &, long double &, long double &);
void writeToFile(const char *, Vector &);
void writeToFile(string &, Vector &);
std::vector<Normal> generateRandomComponentsUnivariate(int);
  
void computeEstimatorsUnivariate(struct Parameters &, Vector &);
void modelOneComponentUnivariate(struct Parameters &, Vector &);
void modelMixtureUnivariate(struct Parameters &, Vector &);
void simulateMixtureModelUnivariate(struct Parameters &);
void compareMixturesUnivariate(struct Parameters &);
void strategic_inference_univariate(struct Parameters &, MixtureUnivariate &, Vector &);
MixtureUnivariate inferComponentsUnivariate(MixtureUnivariate &, int, ostream &);
void updateInference(MixtureUnivariate &, MixtureUnivariate &, int, ostream &, int);

#endif

