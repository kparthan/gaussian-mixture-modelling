#include "MultivariateNormal.h"
#include "Support.h"
#include "Normal.h"

extern int ESTIMATION;

MultivariateNormal::MultivariateNormal()
{
  D = 2;
  mu = Vector(2,0);
  cov = IdentityMatrix(2,2);
  updateConstants();
}

MultivariateNormal::MultivariateNormal(Vector &mu, Matrix &cov) : mu(mu), cov(cov)
{
  D = mu.size();
  updateConstants();
}

void MultivariateNormal::updateConstants()
{
  D = mu.size();
  cov_inv = ZeroMatrix(D,D);
  invertMatrix(cov,cov_inv,det_cov);
  log_cd = computeLogNormalizationConstant();
}

long double MultivariateNormal::computeLogNormalizationConstant()
{
  long double log_c = 0.5 * D * log(2*PI);
  log_c += 0.5 * log(det_cov);
  return -log_c;
}

MultivariateNormal MultivariateNormal::operator=(const MultivariateNormal &source)
{
  if (this != &source) {
    D = source.D;
    mu = source.mu;
    cov = source.cov;
    cov_inv = source.cov_inv;
    det_cov = source.det_cov;
    log_cd = source.log_cd;
  }
  return *this;
}

void MultivariateNormal::printParameters()
{
  cout << "Multivariate Normal Parameters:\n";
  cout << "Mean: "; print(cout,mu,3); cout << endl;
  cout << "Cov: " << cov << endl;
}

void MultivariateNormal::printParameters(ostream &os)
{
  os << "[mu]: "; print(os,mu,3);
  os << "\t[cov]: (";
  for (int i=0; i<D; i++) {
    os << "(";
    for (int j=0; j<D-1; j++) {
      os << fixed << setprecision(3) << cov(i,j) << ", ";
    }
    os << fixed << setprecision(3) << cov(i,D-1) << ")";
  }
  os << ")" << endl;
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
      x[j] = mu[j] + az[j];
    }
    random_sample.push_back(x);
  }
  return random_sample;
}

void MultivariateNormal::computeAllEstimators(
  std::vector<Vector> &data, struct Estimates &estimates, int verbose /* default = 0 (don't print) */
) {
  MultivariateNormal mvnorm_est;
  long double msglen,kldiv;

  Vector weights(data.size(),1.0);

  ESTIMATION = BOTH;
  estimateMean(estimates,data,weights);
  estimateCovariance(estimates,data,weights);

  if (verbose == 1) {
    cout << "mean_est: "; print(cout,estimates.mean,3); cout << endl;

    cout << "\nML:\n";
    cout << "cov_est: " << estimates.cov_ml << endl;
    mvnorm_est = MultivariateNormal(estimates.mean,estimates.cov_ml);
    msglen = mvnorm_est.computeMessageLength(data);
    kldiv = computeKLDivergence(mvnorm_est);
    cout << "msglen: " << msglen << endl;
    cout << "KL-divergence: " << kldiv << endl << endl;

    cout << "\nMML:\n";
    cout << "cov_est: " << estimates.cov_mml << endl;
    mvnorm_est = MultivariateNormal(estimates.mean,estimates.cov_mml);
    msglen = mvnorm_est.computeMessageLength(data);
    kldiv = computeKLDivergence(mvnorm_est);
    cout << "msglen: " << msglen << endl;
    cout << "KL-divergence: " << kldiv << endl << endl;
  }
}

void MultivariateNormal::estimateParameters(std::vector<Vector> &data, Vector &weights)
{
  struct Estimates estimates;

  estimateMean(estimates,data,weights);

  estimateCovariance(estimates,data,weights);

  updateParameters(estimates);
}

void MultivariateNormal::estimateMean(
  struct Estimates &estimates,std::vector<Vector> &data, Vector &weights
) {
  Vector resultant = computeVectorSum(data,weights,estimates.Neff);
  estimates.mean = Vector(resultant.size(),0);
  for (int i=0; i<resultant.size(); i++) {
    estimates.mean[i] = resultant[i] / estimates.Neff;
  }
}

void MultivariateNormal::estimateCovariance(
  struct Estimates &estimates, std::vector<Vector> &data, Vector &weights
) {
  int N = data.size();
  int D = data[0].size();
  Vector emptyvec(D,0);
  std::vector<Vector> x_mu(N,emptyvec);
  for (int i=0; i<N; i++) {
    for (int j=0; j<D; j++) {
      x_mu[i][j] = data[i][j] - estimates.mean[j];
    }
  }

  Matrix S = computeDispersionMatrix(x_mu,weights);
 
  switch(ESTIMATION) {
    case ML:
      estimates.cov_ml = S / estimates.Neff;
      break;

    case MML:
      estimates.cov_mml = S / (estimates.Neff + D - 2);
      break;

    case BOTH:
      estimates.cov_ml = S / estimates.Neff;
      estimates.cov_mml = S / (estimates.Neff + D - 2);
      break;
  }
}

void MultivariateNormal::updateParameters(struct Estimates &estimates)
{
  mu = estimates.mean;
  switch(ESTIMATION) {
    case ML:
      cov = estimates.cov_ml;
      break;

    case MML:
      cov = estimates.cov_mml;
      break;
  }
  updateConstants();
}

Vector MultivariateNormal::Mean()
{
  return mu;
}

Matrix MultivariateNormal::Covariance()
{
  return cov;
}

Matrix MultivariateNormal::CovarianceInverse()
{
  return cov_inv;
}

long double MultivariateNormal::getLogNormalizationConstant()
{
  return log_cd;
}

long double MultivariateNormal::log_density(Vector &x)
{
  long double log_pdf = log_cd;
  Vector diff(D,0);
  for (int i=0; i<D; i++) {
    diff[i] = x[i] - mu[i];
  }
  log_pdf -= (0.5 * prod_vMv(diff,cov_inv));
  return log_pdf;
}

long double MultivariateNormal::computeNegativeLogLikelihood(Vector &x)
{
  long double log_pdf = log_cd;
  Vector diff(D,0);
  for (int i=0; i<D; i++) {
    diff[i] = x[i] - mu[i];
  }
  log_pdf -= (0.5 * prod_vMv(diff,cov_inv));
  return -log_pdf;
}

long double MultivariateNormal::computeNegativeLogLikelihood(std::vector<Vector> &sample)
{
  int N = sample.size();
  long double value = N * log_cd;
  Vector diff(D,0);
  long double tmp = 0;
  for (int i=0; i<N; i++) {
    for (int j=0; j<D; j++) {
      diff[j] = sample[i][j] - mu[j];
    }
    tmp += prod_vMv(diff,cov_inv);
  }
  value -= 0.5 * tmp;
  return -value;
}

long double MultivariateNormal::computeMessageLength(std::vector<Vector> &data)
{
  // msg length to encode parameters
  // (this includes Fisher term as well)
  long double It = computeLogParametersProbability(data.size());

  // msg length to encode data given parameters 
  long double Il = computeNegativeLogLikelihood(data);
  Il -= (D * data.size() * log(AOM));

  int num_params = 0.5 * D * (D+3);
  long double constant = computeConstantTerm(num_params);

  return (It + Il + constant) / log(2);
}

long double MultivariateNormal::computeLogParametersProbability(long double Neff)
{
  long double log_prior_density = computeLogPriorDensity();
  long double log_expected_fisher = computeLogFisherInformation(Neff);
  long double logp = -log_prior_density + 0.5 * log_expected_fisher;
  return logp;
}

long double MultivariateNormal::computeLogPriorDensity()
{
  long double log_prior = D * log(R1);
  log_prior += (0.5 * (D+1) * log(det_cov));
  return -log_prior;
}

long double MultivariateNormal::computeLogFisherInformation(long double Neff)
{
  long double log_fisher = 0;
  log_fisher += (2 * D * log(Neff));
  log_fisher -= (D * log(2));
  log_fisher -= (3 * log(det_cov));
  return log_fisher;
}

long double MultivariateNormal::computeKLDivergence(MultivariateNormal &other)
{
  long double log_cd2 = other.getLogNormalizationConstant();
  long double kldiv = log_cd - log_cd2;

  long double tmp = 0;
  Vector mu2 = other.Mean();
  Matrix cov2_inv = other.CovarianceInverse();
  Vector diff_mu(D,0);
  Matrix x = prod(cov2_inv,cov);
  for (int i=0; i<D; i++) { // trace(c2inv * c1)
    tmp += x(i,i);
    diff_mu[i] = mu2[i] - mu[i];
  }
  tmp += prod_vMv(diff_mu,cov2_inv);
  tmp -= D;
  kldiv += (0.5 * tmp);
  return kldiv / log(2);
}

