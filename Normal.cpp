#include "Normal.h"
//#include "Support.h"
#include "SupportUnivariate.h"

extern int ESTIMATION;

/*!
 *  \brief This is a constructor module
 *  sets default values of mean as 0 and standard deviation as 1
 */
Normal::Normal() : mu(0), sigma(1)
{
  updateConstants();
}

/*!
 *  \brief constructor function which sets the value of mean and 
 *  standard deviation of the distribution
 *  \param mu a long double
 *  \param sigma a long double
 */
Normal::Normal(long double mu, long double sigma) : mu(mu), sigma(sigma)
{
  updateConstants();
}

void Normal::updateConstants()
{
  log_cd = computeLogNormalizationConstant();
}

long double Normal::computeLogNormalizationConstant()
{
  long double log_c = 0.5 * log(2*PI);
  log_c += log(sigma);
  return -log_c;
}

/*!
 *  \brief This function assigns a source Normal distribution.
 *  \param source a reference to a Normal
 */
Normal Normal::operator=(const Normal &source)
{
  if (this != &source) {
    mu = source.mu;
    sigma = source.sigma;
    log_cd = source.log_cd;
  }
  return *this;
}

void Normal::printParameters()
{
  cout << "Univariate Normal Parameters:\n";
  cout << "Mean: " << mu << endl;
  cout << "Sigma: " << sigma << endl;
}

void Normal::printParameters(ostream &os)
{
  os << "[mu]: " << setprecision(3) << mu;
  os << "\t[sigma]: " << setprecision(3) << sigma << endl;
}

void Normal::printParameters(ostream &os, int set)
{
  os << "[mu]: " << scientific << setprecision(6) << mu;
  os << "\t[sigma]: " << scientific << setprecision(6) << sigma << endl;
}

/*!
 *  \brief This function generates random numbers from the Normal distribution.
 *  \param sample_size an integer
 *  \return the random sample
 */
Vector Normal::generate(int sample_size)
{
  Vector sample(sample_size,0);
  long double u,v,r1,r2,sqroot,arg;

  for (int i=0; i<sample_size; i+=2) {
    repeat:
    u = uniform_random();
    sqroot = sqrt(-2 * log(u));

    v = uniform_random();
    if (fabs(u-v) > TOLERANCE) {   // u != v
      arg = 2 * PI * v;
      r1 = sqroot * cos (arg);
      r2 = sqroot * sin (arg);
      sample[i] = mu + sigma * r1;
      if (i != sample_size-1) {
        sample[i+1] = mu + sigma * r2;
      }
    } else {
      goto repeat;
    }
  }
  return sample;
}

void Normal::computeAllEstimators(
  Vector &data, struct EstimatesUnivariate &estimates, int verbose /* default = 0 (don't print) */
) {
  Normal norm_est;
  long double msglen,kldiv;

  Vector weights(data.size(),1.0);

  ESTIMATION = BOTH;
  estimateMean(estimates,data,weights);
  estimateSigma(estimates,data,weights);

  if (verbose == 1) {
    cout << "mean_est: " << estimates.mean << endl;

    cout << "\nML:\n";
    cout << "sigma_est: " << estimates.sigma_ml << endl;
    norm_est = Normal(estimates.mean,estimates.sigma_ml);
    msglen = norm_est.computeMessageLength(data);
    kldiv = computeKLDivergence(norm_est);
    cout << "msglen: " << msglen << endl;
    cout << "KL-divergence: " << kldiv << endl << endl;

    cout << "\nMML:\n";
    cout << "sigma_est: " << estimates.sigma_mml << endl;
    norm_est = Normal(estimates.mean,estimates.sigma_mml);
    msglen = norm_est.computeMessageLength(data);
    kldiv = computeKLDivergence(norm_est);
    cout << "msglen: " << msglen << endl;
    cout << "KL-divergence: " << kldiv << endl << endl;
  }
}

void Normal::estimateParameters(Vector &data, Vector &weights)
{
  struct EstimatesUnivariate estimates;

  estimateMean(estimates,data,weights);

  estimateSigma(estimates,data,weights);

  updateParameters(estimates);
}

void Normal::estimateMean(
  struct EstimatesUnivariate &estimates, Vector &data, Vector &weights
) {
  long double resultant = computeSum(data,weights,estimates.Neff);
  estimates.mean = resultant / estimates.Neff;
}

void Normal::estimateSigma(
  struct EstimatesUnivariate &estimates, Vector &data, Vector &weights
) {
  int N = data.size();
  long double sum_sqd_dev = 0;
  long double x_mu;
  for (int i=0; i<N; i++) {
    x_mu = data[i] - estimates.mean;
    sum_sqd_dev += (weights[i] * x_mu * x_mu);
  }

  switch(ESTIMATION) {
    case ML:
      estimates.sigma_ml = sqrt(sum_sqd_dev / estimates.Neff);
      break;

    case MML:
      if (estimates.Neff <= 1) {
        cout << "Neff: " << estimates.Neff << " <= 1" << endl;
        estimates.sigma_mml = sqrt(sum_sqd_dev / estimates.Neff);
      } else {
        estimates.sigma_mml = sqrt(sum_sqd_dev / (estimates.Neff - 1));
      }
      break;

    case BOTH:
      estimates.sigma_ml = sqrt(sum_sqd_dev / estimates.Neff);
      estimates.sigma_mml = sqrt(sum_sqd_dev / (estimates.Neff - 1));
      break;
  }
}

void Normal::updateParameters(struct EstimatesUnivariate &estimates)
{
  mu = estimates.mean;
  switch(ESTIMATION) {
    case ML:
      sigma = estimates.sigma_ml;
      break;

    case MML:
      sigma = estimates.sigma_mml;
      break;
  }
  //cout << "Neff: " << estimates.Neff << endl;
  updateConstants();
}

/*!
 *  \brief This function returns the mean of the distribution
 *  \return the mean of the distribution
 */
long double Normal::Mean(void)
{
	return mu;
}

/*!
 *  \brief This function returns the standard deviation of the distribution
 *  \return the standard deviation of the distribution
 */
long double Normal::Sigma(void)
{
	return sigma;
}

long double Normal::getLogNormalizationConstant()
{
  return log_cd;
}

/*!
 *  \brief This function computes the value of the distribution at a given x
 *  \param x a long double
 *  \return density of the function given x
 */
long double Normal::log_density(long double &x)
{
  long double log_pdf = log_cd;
  long double z = (x-mu)/(long double)sigma;
  long double exponent = 0.5 * z * z;
  log_pdf -= exponent;
  assert(!boost::math::isnan(log_pdf));
  return log_pdf;
}

/*!
 *  \brief This function computes the negative log likelihood of given data.
 *  \param sample a reference to a Vector
 *  \return the negative log likelihood (base e)
 */
long double Normal::computeNegativeLogLikelihood(long double &x)
{
  return -log_density(x);
}

/*!
 *  \brief This function computes the negative log likelihood of given data.
 *  \param sample a reference to a Vector
 *  \return the negative log likelihood (base e)
 */
long double Normal::computeNegativeLogLikelihood(Vector &sample)
{
  long double value = sample.size() * log_cd;

  long double num = 0;
  for (int i=0; i<sample.size(); i++) {
    num += (sample[i]-mu) * (sample[i]-mu);
  }
  long double denom = 2 * sigma * sigma;
  value -= (num / denom);
  return -value;
}

long double Normal::computeMessageLength(Vector &data)
{
  // msg length to encode parameters
  // (this includes Fisher term as well)
  long double It = computeLogParametersProbability(data.size());

  // msg length to encode data given parameters 
  long double Il = computeNegativeLogLikelihood(data);
  Il -= (data.size() * log(AOM));

  int num_params = 2;
  long double constant = computeConstantTerm(num_params);

  return (It + Il + constant) / log(2);
}

long double Normal::computeLogParametersProbability(long double Neff)
{
  long double log_prior_density = computeLogPriorDensity();
  long double log_expected_fisher = computeLogFisherInformation(Neff);
  long double logp = -log_prior_density + 0.5 * log_expected_fisher;
  return logp;
}

long double Normal::computeLogPriorDensity()
{
  long double log_prior = log(R1);
  log_prior += log(R2) ;
  log_prior += log(sigma);
  return -log_prior;
}

long double Normal::computeLogFisherInformation(long double Neff)
{
  long double log_fisher = log(2);
  log_fisher += (2 * log(Neff));
  log_fisher -= (4 * log(sigma));
  return log_fisher;
}

long double Normal::entropy()
{
  long double ans = 1 + log(2*PI);
  ans *= 0.5;
  ans += log(sigma);
  return ans;
}

long double Normal::computeKLDivergence(Normal &other)
{
  long double mu2 = other.Mean();
  long double sigma2 = other.Sigma();

  long double kldiv = log(sigma2) - log(sigma);
  kldiv -= 0.5;

  long double diff = mu - mu2;
  long double tmp = (sigma * sigma) + (diff * diff);
  tmp /= (2 * sigma2 * sigma2);
  kldiv += tmp;

  return kldiv;
}

Normal Normal::conflate(Normal &other)
{
  long double mu2 = other.Mean();
  long double sigma2 = other.Sigma();

  long double inv1 = 1.0 / (sigma * sigma);
  long double inv2 = 1.0 / (sigma2 * sigma2);
  long double denom = inv1 + inv2;
  long double var = 1 / denom;
  long double sigma3 = sqrt(var);

  long double tmp1 = mu * inv1;
  long double tmp2 = mu2 * inv2;
  long double num = tmp1 + tmp2;
  long double mu3 = num / denom;

  return Normal(mu3,sigma3);
}

