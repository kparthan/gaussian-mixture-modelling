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
  assert(verify(cov) == 1);
  updateConstants();
}

void MultivariateNormal::updateConstants()
{
  D = mu.size();
  cov_inv = ZeroMatrix(D,D);
  invertMatrix(cov,cov_inv,det_cov);
  if (det_cov < 0) {
    cout << "cov: " << cov << endl;
    cout << "inverse: " << cov_inv << endl;
    cout << "det_cov: " << det_cov << endl;
    /*if (fabs(det_cov) < 1e-6) {
      det_cov = fabs(det_cov);
    }*/
  }
  assert(det_cov > 0);
  det_cov = fabs(det_cov);

  /*long double MIN_SIGMA = AOM;
  if (det_cov < 0 || fabs(det_cov) < 1e-12) {
    long double min_var = MIN_SIGMA * MIN_SIGMA;
    Matrix I = IdentityMatrix(D,D);
    cov = min_var * I;
    cov_inv = I / min_var;
    det_cov = min_var * min_var;
  }*/
  /*if (det_cov < 0) {
    cout << "cov: " << cov << endl;
    cout << "det_cov: " << det_cov << endl;
  }
  assert(det_cov > 0);*/

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

void MultivariateNormal::printParameters(ostream &os, int set)
{
  os << "[mu]: "; print(os,mu,3);
  os << "\t[cov]: (";
  os << fixed << setprecision(6); 
  for (int i=0; i<D; i++) {
    os << "(";
    for (int j=0; j<D-1; j++) {
      os << scientific << cov(i,j) << ", ";
    }
    os << scientific << cov(i,D-1) << ")";
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
      estimates.cov_mml = S / (estimates.Neff - 1);
      break;

    case BOTH:
      estimates.cov_ml = S / estimates.Neff;
      estimates.cov_mml = S / (estimates.Neff - 1);
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

int MultivariateNormal::getDimensionality()
{
  return D;
}

long double MultivariateNormal::log_density(Vector &x)
{
  long double log_pdf = log_cd;
  Vector diff(D,0);
  for (int i=0; i<D; i++) {
    diff[i] = x[i] - mu[i];
  }
  log_pdf -= (0.5 * prod_vMv(diff,cov_inv));
  assert(!boost::math::isnan(log_pdf));
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
  // prior on mu
  long double log_prior = D * log(R1);

  // prior on cov (inverse wishart assumption)
  /*long double df = D+2; // > D - 1 (real)
  log_prior += (0.5 * df * D * log(2)); // constant term
  log_prior += computeLogMultivariateGamma(D,0.5*df);
  // trace of cov_inv
  long double trace = 0;
  for (int i=0; i<D; i++) {
    trace += cov_inv(i,i);
  }
  assert(trace > 0);
  log_prior += 0.5 * trace;
  log_prior += (0.5 * (df+D+1) * log(det_cov));*/
  log_prior += (0.5 * (D+1) * log(det_cov));
  return -log_prior;
}

long double MultivariateNormal::computeLogFisherInformation(long double Neff)
{
  int dim = 0.5 * D * (D+1);
  Matrix cov_inv_fisher = ZeroMatrix(dim,dim);
  std::vector<std::vector<TwoPairs> > pairs = generatePairs(D);
  TwoPairs instance;

  long double v_ik,v_jl,v_il,v_jk;
  int i,j,k,l;
  for (int row=0; row<dim; row++) {
    for (int col=0; col<dim; col++) {
      instance = pairs[row][col];
      i = instance.p1[0];
      j = instance.p1[1];
      k = instance.p2[0];
      l = instance.p2[1];
      v_ik = cov(i,k);
      v_jl = cov(j,l);
      v_il = cov(i,l);
      v_jk = cov(j,k);
      cov_inv_fisher(row,col) = v_ik * v_jl + v_il * v_jk;
    } // col
  } // row
  Matrix cov_fisher(dim,dim);
  long double det_cov_inv_fisher;
  invertMatrix(cov_inv_fisher,cov_fisher,det_cov_inv_fisher);
  //assert(det_cov_inv_fisher > 0);
  if (det_cov_inv_fisher <= 0) {
    cout << "det_cov_inv_fisher: " << det_cov_inv_fisher << endl;
    det_cov_inv_fisher *= -1;
  }

  long double log_fisher = 0;
  log_fisher += (0.5 * D * (D+3) * log(Neff));
  log_fisher -= log(det_cov); // mu fisher term
  log_fisher -= log(det_cov_inv_fisher);  // cov fisher term
  return log_fisher;
}

long double MultivariateNormal::entropy()
{
  long double ans = 1 + log(2*PI);
  ans *= D;
  ans += log(det_cov);
  ans *= 0.5;
  return ans;
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
  return kldiv;
}

MultivariateNormal MultivariateNormal::conflate(MultivariateNormal &other)
{
  Vector mu2 = other.Mean();
  Matrix cov2 = other.Covariance();

  Matrix sum = cov + cov2;
  Matrix sum_inv = ZeroMatrix(D,D);
  long double det = 1;
  invertMatrix(sum,sum_inv,det);

  Matrix tmp;
  tmp = prod(cov,sum_inv);
  Matrix cov3 = prod(tmp,cov2);

  Vector tmp2 = prod(sum_inv,mu);
  Vector tmp3 = prod(cov2,tmp2);
  Vector tmp4 = prod(sum_inv,mu2);
  Vector tmp5 = prod(cov,tmp4);
  Vector mu3(D,0);
  for (int i=0; i<D; i++) {
    mu3[i] = tmp3[i] + tmp5[i];
  }
  return MultivariateNormal(mu3,cov3);
}

