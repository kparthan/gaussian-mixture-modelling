#include "Experiments.h"
#include "MultivariateNormal.h"
#include "Support.h"

Experiments::Experiments(int iterations) : iterations(iterations)
{}

void Experiments::simulate(int D)
{
  int N = 10000;

  Vector mean(D,0);
  Matrix cov = IdentityMatrix(D,D);
  MultivariateNormal mvnorm(mean,cov),mvnorm_est;

  string D_str = boost::lexical_cast<string>(D);
  string size_str = boost::lexical_cast<string>(N);
  string folder = "./experiments/D_" + D_str + "/";
  string negloglkhd_file = folder + "n_" + size_str + "_negloglikelihood";
  string kldvg_file = folder + "n_" + size_str + "_kldiv";
  string msglens_file = folder + "n_" + size_str + "_msglens";
  ofstream logneg(negloglkhd_file.c_str(),ios::app);
  ofstream logkldiv(kldvg_file.c_str(),ios::app);
  ofstream logmsg(msglens_file.c_str(),ios::app);

  Vector emptyvec(2,0);
  std::vector<Vector> negloglkhd(iterations,emptyvec),kldiv(iterations,emptyvec),msglens(iterations,emptyvec);
  long double actual_negloglkhd,actual_msglen;
  for (int iter=0; iter<iterations; iter++) {
    std::vector<Vector> data = mvnorm.generate(N);
    struct Estimates estimates;
    mvnorm_est = MultivariateNormal(mean,cov);
    mvnorm_est.computeAllEstimators(data,estimates);

    actual_negloglkhd = mvnorm.computeNegativeLogLikelihood(data) / log(2);
    actual_msglen = mvnorm.computeMessageLength(data);
    logneg << scientific << actual_negloglkhd << "\t";
    logmsg << scientific << actual_msglen << "\t";

    // ML
    MultivariateNormal fit(estimates.mean,estimates.cov_ml);
    negloglkhd[iter][0] = fit.computeNegativeLogLikelihood(data) / log(2);
    msglens[iter][0] = fit.computeMessageLength(data);
    kldiv[iter][0] = mvnorm.computeKLDivergence(fit);
    logneg << scientific << negloglkhd[iter][0] << "\t";
    logmsg << scientific << msglens[iter][0] << "\t";
    logkldiv << scientific << kldiv[iter][0] << "\t";

    // MML
    fit = MultivariateNormal(estimates.mean,estimates.cov_mml);
    negloglkhd[iter][1] = fit.computeNegativeLogLikelihood(data) / log(2);
    msglens[iter][1] = fit.computeMessageLength(data);
    kldiv[iter][1] = mvnorm.computeKLDivergence(fit);
    logneg << scientific << negloglkhd[iter][1] << "\t";
    logmsg << scientific << msglens[iter][1] << "\t";
    logkldiv << scientific << kldiv[iter][1] << "\t";
    
    logneg << endl; logmsg << endl; logkldiv << endl; 
  }
  logneg.close(); logmsg.close(); logkldiv.close();
}

