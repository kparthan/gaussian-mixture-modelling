#include "Normal.h"
#include "Support.h"

/*!
 *  \brief This is a constructor module
 *  sets default values of mean as 0 and standard deviation as 1
 */
Normal::Normal() : mu(0), sigma(1)
{}

/*!
 *  \brief constructor function which sets the value of mean and 
 *  standard deviation of the distribution
 *  \param mu a long double
 *  \param sigma a long double
 */
Normal::Normal(long double mu, long double sigma) : mu(mu), sigma(sigma)
{}

/*!
 *  \brief This function assigns a source Normal distribution.
 *  \param source a reference to a Normal
 */
Normal Normal::operator=(const Normal &source)
{
  if (this != &source) {
    mu = source.mu;
    sigma = source.sigma;
  }
  return *this;
}

/*!
 *  \brief This function returns the mean of the distribution
 *  \return the mean of the distribution
 */
const long double Normal::mean(void)
{
	return mu;
}

/*!
 *  \brief This function returns the standard deviation of the distribution
 *  \return the standard deviation of the distribution
 */
const long double Normal::standardDeviation(void)
{
	return sigma;
}

/*!
 *  \brief This function computes the value of the distribution at a given x
 *  \param x a long double
 *  \return density of the function given x
 */
long double Normal::density(long double x)
{
  long double tmp = (x-mu)/(long double)sigma;
  long double exponent = -0.5 * tmp * tmp;
	long double val = exp(exponent) / ((sqrt(2*PI)) * sigma);
  return val;
}

/*!
 *  \brief This function computes the negative log likelihood of given data.
 *  \param sample a reference to a Vector
 *  \return the negative log likelihood (base e)
 */
long double Normal::negativeLogLikelihood(long double x)
{
  long double pdf = density(x);
  return -log(pdf);
}

/*!
 *  \brief This function computes the negative log likelihood of given data.
 *  \param sample a reference to a Vector
 *  \return the negative log likelihood (base e)
 */
long double Normal::negativeLogLikelihood(Vector &sample)
{
  long double value = 0;
  long double tmp = 0.5 * log(2*PI) + log(sigma);
  value += (sample.size() * tmp);

  long double num = 0;
  for (int i=0; i<sample.size(); i++) {
    num += (sample[i]-mu) * (sample[i]-mu);
  }
  long double denom = 2 * sigma * sigma;
  value += num / (long double)denom;
  return value;
}

/*!
 *  \brief This function prints the parameters of the model.
 */
void Normal::printParameters(ostream &os)
{
  os << "Mean (mu): " << mu << endl;
  os << "Standard deviation (sigma): " << sigma << endl;
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

