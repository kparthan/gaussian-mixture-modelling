#include "Mixture.h"
#include "Support.h"

extern int MIXTURE_ID;
extern int MIXTURE_SIMULATION;
extern int INFER_COMPONENTS;
extern int ENABLE_DATA_PARALLELISM;
extern int NUM_THREADS;
extern long double IMPROVEMENT_RATE;
extern int ESTIMATION;
extern int TOTAL_ITERATIONS;
int SPLITTING = 0;
extern int IGNORE_SPLIT;
extern long double MIN_N;

long double IK,IW,IL,sum_IT;
Vector IT;

/*!
 *  \brief Null constructor module
 */
Mixture::Mixture()
{
  id = MIXTURE_ID++;
}

/*!
 *  \brief This is a constructor function which instantiates a Mixture
 *  \param K an integer
 *  \param components a reference to a std::vector<MultivariateNormal>
 *  \param weights a reference to a Vector 
 */
Mixture::Mixture(int K, std::vector<MultivariateNormal> &components, Vector &weights):
                 K(K), components(components), weights(weights)
{
  assert(components.size() == K);
  assert(weights.size() == K);
  id = MIXTURE_ID++;
  minimum_msglen = 0;
}

/*!
 *  \brief This is a constructor function.
 *  \param K an integer
 *  \param data a reference to a std::vector<Vector>
 *  \param data_weights a reference to a Vector
 */
Mixture::Mixture(int K, std::vector<Vector> &data, Vector &data_weights) : 
                 K(K), data(data), data_weights(data_weights)
{
  id = MIXTURE_ID++;
  N = data.size();
  assert(data_weights.size() == N);
  minimum_msglen = 0;
}

/*!
 *  \brief This is a constructor function.
 *  \param K an integer
 *  \param components a reference to a std::vector<MultivariateNormal>
 *  \param weights a reference to a Vector
 *  \param sample_size a reference to a Vector
 *  \param responsibility a reference to a std::vector<Vector>
 *  \param data a reference to a std::vector<Vector>
 *  \param data_weights a reference to a Vector
 */
Mixture::Mixture(
  int K, 
  std::vector<MultivariateNormal> &components, 
  Vector &weights,
  Vector &sample_size, 
  std::vector<Vector> &responsibility,
  std::vector<Vector> &data,
  Vector &data_weights
) : K(K), components(components), weights(weights), sample_size(sample_size),
    responsibility(responsibility), data(data), data_weights(data_weights)
{
  id = MIXTURE_ID++;
  assert(components.size() == K);
  assert(weights.size() == K);
  assert(sample_size.size() == K);
  assert(responsibility.size() == K);
  N = data.size();
  assert(data_weights.size() == N);
  minimum_msglen = 0;
}

/*!
 *  \brief This function assigns a source Mixture distribution.
 *  \param source a reference to a Mixture
 */
Mixture Mixture::operator=(const Mixture &source)
{
  if (this != &source) {
    id = source.id;
    N = source.N;
    K = source.K;
    components = source.components;
    data = source.data;
    data_weights = source.data_weights;
    responsibility = source.responsibility;
    sample_size = source.sample_size;
    weights = source.weights;
    msglens = source.msglens;
    null_msglen = source.null_msglen;
    minimum_msglen = source.minimum_msglen;
    part1 = source.part1;
    part2 = source.part2;
  }
  return *this;
}

/*!
 *  \brief This function checks whether the two Mixture objects are the same.
 *  \param other a reference to a Mixture
 *  \return whether they are the same object or not
 */
bool Mixture::operator==(const Mixture &other)
{
  if (id == other.id) {
    return 1;
  } else {
    return 0;
  }
}

/*!
 *  \brief This function returns the list of all weights.
 *  \return the list of weights
 */
Vector Mixture::getWeights()
{
  return weights;
}

/*!
 *  \brief This function returns the list of components.
 *  \return the components
 */
std::vector<MultivariateNormal> Mixture::getComponents()
{
  return components;
}

/*!
 *  \brief Gets the number of components
 */
int Mixture::getNumberOfComponents()
{
  return components.size();
}

/*!
 *  \brief This function returns the responsibility matrix.
 */
std::vector<Vector> Mixture::getResponsibilityMatrix()
{
  return responsibility;
}

/*!
 *  \brief This function returns the sample size of the mixture.
 *  \return the sample size
 */
Vector Mixture::getSampleSize()
{
  return sample_size;
}

/*!
 *  \brief This function initializes the parameters of the model.
 */
void Mixture::initialize()
{
  N = data.size();
  cout << "Sample size: " << N << endl;

  // initialize responsibility matrix
  //srand(time(NULL));
  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);

  //#pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) 
  /*for (int i=0; i<N; i++) {
    int index = rand() % K;
    responsibility[index][i] = 1;
  }*/
  /*for (int i=0; i<N; i++) {
    for (int j=0; j<K; j++)
    responsibility[j][i] = uniform_random();
  }*/
  /*for (int i=0; i<N; i++) {
    if (K == 2) {
      if (i < 0.5 * N) {
        responsibility[0][i] = 1;
      } else {
        responsibility[1][i] = 1;
      }
    } else {
    int index = rand() % K;
    responsibility[index][i] = 1;}
  }*/
  for (int i=0; i<N; i++) {
    long double sum = 0;
    for (int j=0; j<K; j++) {
      long double random = uniform_random();
      responsibility[j][i] = random;
      sum += random;
    }
    for (int j=0; j<K; j++) {
      responsibility[j][i] /= sum;
    }
  }
  //writeToFile("resp",responsibility,3); exit(1);
  sample_size = Vector(K,0);
  updateEffectiveSampleSize();
  weights = Vector(K,0);
  if (ESTIMATION == MML) {
    updateWeights();
  } else {
    updateWeights_ML();
  }

  // initialize parameters of each component
  components = std::vector<MultivariateNormal>(K);
  updateComponents();
}

void Mixture::initialize2()
{
  N = data.size();
  int D = data[0].size();
  cout << "Sample size: " << N << endl;

  long double init_weight = 1.0 / K;
  weights = Vector(K,init_weight);

  // determine covariance of the data
  Vector global_mean;
  Matrix global_cov;
  Vector data_weights(N,1);
  computeMeanAndCovariance(data,data_weights,global_mean,global_cov);
  Matrix cov = 0.1 * global_cov;

  // choose K random means by choosing K random points
  std::vector<int> flags(N,0);
  for (int i=0; i<K; i++) {
    int index = rand() % N;
    if (flags[index] == 0) {
      MultivariateNormal mvnorm(data[index],cov);
      components.push_back(mvnorm);
      flags[index] = 1;
    } else i--;
  }

  // init sample_size and responsibility variables
  sample_size = Vector(K,0);
  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);
  updateResponsibilityMatrix();
  updateEffectiveSampleSize();
  updateWeights();
  updateComponents();
}

void Mixture::initialize3()
{
  N = data.size();
  int D = data[0].size();
  cout << "Sample size: " << N << endl;

  // choose K random means by choosing K random points
  std::vector<Vector> init_means(K);
  std::vector<int> flags(N,0);
  for (int i=0; i<K; i++) {
    int index = rand() % N;
    if (flags[index] == 0) {
      init_means[i] = data[i];
      flags[index] = 1;
    } else i--;
  }

  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);
  long double dist,min_dist;
  int nearest;
  for (int i=0; i<N; i++) {
    min_dist = computeEuclideanDistance(init_means[0],data[i]);
    nearest = 0;
    for (int j=1; j<K; j++) {
      dist = computeEuclideanDistance(init_means[j],data[i]);
      if (dist < min_dist) {
        min_dist = dist;
        nearest = j;
      }
    } // for j()
    responsibility[nearest][i] = 1;
  } // for i()

  sample_size = Vector(K,0);
  updateEffectiveSampleSize();
  weights = Vector(K,0);
  if (ESTIMATION == MML) {
    updateWeights();
  } else {
    updateWeights_ML();
  }

  // initialize parameters of each component
  components = std::vector<MultivariateNormal>(K);
  updateComponents();
}

/*!
 *  \brief This function updates the effective sample size of each component.
 */
void Mixture::updateEffectiveSampleSize()
{
  for (int i=0; i<K; i++) {
    long double count = 0;
    #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) reduction(+:count)
    for (int j=0; j<N; j++) {
      count += responsibility[i][j];
    }
    sample_size[i] = count;
  }
}

void Mixture::updateEffectiveSampleSize(int index)
{
  //for (int i=0; i<K; i++) {
    long double count = 0;
    #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) reduction(+:count)
    for (int j=0; j<N; j++) {
      count += responsibility[index][j];
    }
    sample_size[index] = count;
  //}
}

/*!
 *  \brief This function is used to update the weights of the components.
 */
void Mixture::updateWeights()
{
  long double normalization_constant = N + (K/2.0);
  for (int i=0; i<K; i++) {
    weights[i] = (sample_size[i] + 0.5) / normalization_constant;
  }
}

void Mixture::updateWeights(int index)
{
  long double normalization_constant = N + (K/2.0);
  //for (int i=0; i<K; i++) {
    weights[index] = (sample_size[index] + 0.5) / normalization_constant;
  //}
}

void Mixture::updateWeights_ML()
{
  long double normalization_constant = N;
  for (int i=0; i<K; i++) {
    weights[i] = sample_size[i] / normalization_constant;
  }
}

void Mixture::updateWeights_ML(int index)
{
  long double normalization_constant = N;
  //for (int i=0; i<K; i++) {
    weights[index] = sample_size[index] / normalization_constant;
  //}
}

/*!
 *  \brief This function is used to update the components.
 */
void Mixture::updateComponents()
{
  Vector comp_data_wts(N,0);
  for (int i=0; i<K; i++) {
    #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) 
    for (int j=0; j<N; j++) {
      comp_data_wts[j] = responsibility[i][j] * data_weights[j];
    }
    components[i].estimateParameters(data,comp_data_wts);
    //components[i].updateParameters();
  }
}

void Mixture::updateComponents(int index)
{
  Vector comp_data_wts(N,0);
  //for (int i=0; i<K; i++) {
    #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) 
    for (int j=0; j<N; j++) {
      comp_data_wts[j] = responsibility[index][j] * data_weights[j];
    }
    components[index].estimateParameters(data,comp_data_wts);
    //components[i].updateParameters();
  //}
}

/*!
 *  \brief This function updates the terms in the responsibility matrix.
 */
void Mixture::updateResponsibilityMatrix()
{
  #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) //private(j)
  for (int i=0; i<N; i++) {
    Vector log_densities(K,0);
    for (int j=0; j<K; j++) {
      log_densities[j] = components[j].log_density(data[i]);
    }
    int max_index = maximumIndex(log_densities);
    long double max_log_density = log_densities[max_index];
    for (int j=0; j<K; j++) {
      log_densities[j] -= max_log_density; 
    }
    long double px = 0;
    Vector probabilities(K,0);
    for (int j=0; j<K; j++) {
      probabilities[j] = weights[j] * exp(log_densities[j]);
      px += probabilities[j];
    }
    for (int j=0; j<K; j++) {
      responsibility[j][i] = probabilities[j] / px;
      if(boost::math::isnan(responsibility[j][i])) {
        cout << "i: " << i << "; j: " << j << endl;
        cout << "probs: "; print(cout,probabilities,3); cout << endl;
        writeToFile("resp",responsibility,3); //exit(1);
        printParameters(cout,0,0); 
      }
      assert(!boost::math::isnan(responsibility[j][i]));
    }
  }
}

void Mixture::updateResponsibilityMatrix(int index)
{
  #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) //private(j)
  for (int i=0; i<N; i++) {
    Vector log_densities(K,0);
    for (int j=0; j<K; j++) {
      log_densities[j] = components[j].log_density(data[i]);
    }
    int max_index = maximumIndex(log_densities);
    long double max_log_density = log_densities[max_index];
    for (int j=0; j<K; j++) {
      log_densities[j] -= max_log_density; 
    }
    long double px = 0;
    Vector probabilities(K,0);
    for (int j=0; j<K; j++) {
      probabilities[j] = weights[j] * exp(log_densities[j]);
      px += probabilities[j];
    }
    //for (int j=0; j<K; j++) {
      responsibility[index][i] = probabilities[index] / px;
      assert(!boost::math::isnan(responsibility[index][i]));
    //}
  }
}

/*!
 *  \brief This function updates the terms in the responsibility matrix.
 */
void Mixture::computeResponsibilityMatrix(std::vector<Vector> &sample,
                                          string &output_file)
{
  int sample_size = sample.size();
  Vector tmp(sample_size,0);
  std::vector<Vector> resp(K,tmp);
  #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) //private(j)
  for (int i=0; i<sample_size; i++) {
    Vector log_densities(K,0);
    for (int j=0; j<K; j++) {
      log_densities[j] = components[j].log_density(sample[i]);
    }
    int max_index = maximumIndex(log_densities);
    long double max_log_density = log_densities[max_index];
    for (int j=0; j<K; j++) {
      log_densities[j] -= max_log_density; 
    }
    long double px = 0;
    Vector probabilities(K,0);
    for (int j=0; j<K; j++) {
      probabilities[j] = weights[j] * exp(log_densities[j]);
      px += probabilities[j];
    }
    for (int j=0; j<K; j++) {
      resp[j][i] = probabilities[j] / px;
    }
  }
  ofstream out(output_file.c_str());
  for (int i=0; i<sample_size; i++) {
    for (int j=0; j<K; j++) {
      out << fixed << setw(10) << setprecision(5) << resp[j][i];
    }
    out << endl;
  }
  out.close();
}

/*!
 *
 */
long double Mixture::log_probability(Vector &x)
{
  Vector log_densities(K,0);
  for (int j=0; j<K; j++) {
    log_densities[j] = components[j].log_density(x);
    //assert(!boost::math::isnan(log_densities[j]));
    //assert(log_densities[j] < 0);
  }
  int max_index = maximumIndex(log_densities);
  long double max_log_density = log_densities[max_index];
  for (int j=0; j<K; j++) {
    log_densities[j] -= max_log_density;
  }
  long double density = 0;
  for (int j=0; j<K; j++) {
    density += weights[j] * exp(log_densities[j]);
  }
  return max_log_density + log(density);
}

/*!
 *  \brief This function computes the negative log likelihood of a data
 *  sample.
 *  \param a reference to a std::vector<array<long double,2> >
 *  \return the negative log likelihood (base e)
 */
long double Mixture::negativeLogLikelihood(std::vector<Vector> &sample)
{
  long double value = 0,log_density;
  #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) private(log_density) reduction(-:value)
  for (int i=0; i<sample.size(); i++) {
    log_density = log_probability(sample[i]);
    if(boost::math::isnan(log_density)) {
      writeToFile("resp",responsibility,3); 
    }
    value -= log_density;
  }
  return value;
}

/*!
 *  \brief This function computes the minimum message length using the current
 *  model parameters.
 *  \return the minimum message length
 */
long double Mixture::computeMinimumMessageLength()
{
  IT.clear();
  // encode the number of components
  // assume uniform priors
  long double Ik = log(MAX_COMPONENTS);
  cout << "Ik: " << Ik << endl;
  assert(Ik > 0);

  // enocde the weights
  long double Iw = ((K-1)/2.0) * log(N);
  Iw -= boost::math::lgamma<long double>(K); // log(K-1)!
  for (int i=0; i<K; i++) {
    Iw -= 0.5 * log(weights[i]);
  }
  cout << "Iw: " << Iw << endl;
  //assert(Iw >= 0);

  // encode the likelihood of the sample
  int D = data[0].size();
  long double Il_partial = negativeLogLikelihood(data);
  long double Il = Il_partial - (D * N * log(AOM));
  cout << "Il: " << Il << endl;
  //assert(Il > 0);
  if (Il < 0) {
    minimum_msglen = LARGE_NUMBER;
    return minimum_msglen;
  }

  // encode the parameters of the components
  long double It = 0,logp;
  /*for (int i=0; i<K; i++) {
    logp = components[i].computeLogParametersProbability(sample_size[i]);
    IT.push_back(logp);
    It += logp;
  }
  cout << "It: " << It << endl;*/
  /*if (It <= 0) { cout << It << endl;}
  fflush(stdout);
  assert(It > 0);*/

  // the constant term
  int num_free_params = (0.5 * D * (D+3) * K) + (K - 1);
  long double cd = computeConstantTerm(num_free_params);
  cout << "cd: " << cd << endl;

  minimum_msglen = (Ik + Iw + Il + It + cd)/(log(2));

  part2 = Il + num_free_params/2.0;
  part2 /= log(2);
  part1 = minimum_msglen - part2;

  IK = Ik; IW = Iw; IL = Il; sum_IT = It;

  return minimum_msglen;
}

/*!
 *  \brief Prepares the appropriate log file
 */
string Mixture::getLogFile()
{
  string file_name;
  if (INFER_COMPONENTS == UNSET) {
    if (MIXTURE_SIMULATION == UNSET) {
      file_name = "./mixture/logs/";
    } else if (MIXTURE_SIMULATION == SET) {
      file_name = "./simulation/logs/";
    }
  } else if (INFER_COMPONENTS == SET) {
    file_name = "./infer/logs/";
    file_name += "m_" + boost::lexical_cast<string>(id) + "_";
  }
  file_name += boost::lexical_cast<string>(K) + ".log";
  return file_name;
}

/*!
 *  \brief This function is used to estimate the model parameters by running
 *  an EM algorithm.
 *  \return the stable message length
 */
long double Mixture::estimateParameters()
{
  /*if (SPLITTING == 1) {
    initialize();
  } else {
    initialize2();
  }*/

  //initialize();

  initialize2();

  //initialize3();

  EM();

  //CEM();

  return minimum_msglen;
}

/*!
 *  \brief This function runs the EM method.
 */
void Mixture::EM()
{
  /* prepare log file */
  string log_file = getLogFile();
  ofstream log(log_file.c_str());

  long double prev=0,current;
  int iter = 1;
  printParameters(log,0,0);

  /* EM loop */
  if (ESTIMATION == MML) {
    while (1) {
      // Expectation (E-step)
      updateResponsibilityMatrix();
      updateEffectiveSampleSize();
      if (SPLITTING == 1) {
        for (int i=0; i<K; i++) {
          if (sample_size[i] < MIN_N) {
            current = computeMinimumMessageLength();
            goto stop;
          }
        }
      }
      // Maximization (M-step)
      updateWeights();
      updateComponents();
      current = computeMinimumMessageLength();
      if (fabs(current) >= INFINITY) break;
      msglens.push_back(current);
      printParameters(log,iter,current);
      if (iter != 1) {
        assert(current > 0);
        // because EM has to consistently produce lower 
        // message lengths otherwise something wrong!
        // IMPORTANT: the below condition should not be 
        //          fabs(prev - current) <= 0.0001 * fabs(prev)
        // ... it's very hard to satisfy this condition and EM() goes into
        // ... an infinite loop!
        if (iter > 10 && (prev - current) <= IMPROVEMENT_RATE * prev) {
          stop:
          log << "\nSample size: " << N << endl;
          log << "encoding rate: " << current/N << " bits/point" << endl;
          break;
        }
      }
      prev = current;
      iter++;
      TOTAL_ITERATIONS++;
    }
  } else if (ESTIMATION == ML) {  // ESTIMATION != MML
    while (1) {
      // Expectation (E-step)
      updateResponsibilityMatrix();
      updateEffectiveSampleSize();
      // Maximization (M-step)
      updateWeights_ML();
      updateComponents();
      //current = negativeLogLikelihood(data);
      current = computeMinimumMessageLength();
      msglens.push_back(current);
      printParameters(log,iter,current);
      if (iter != 1) {
        //assert(current > 0);
        // because EM has to consistently produce lower 
        // -ve likelihood values otherwise something wrong!
        if (iter > 10 && (prev - current) <= IMPROVEMENT_RATE * prev) {
          current = computeMinimumMessageLength();
          log << "\nSample size: " << N << endl;
          log << "encoding rate (using ML): " << current/N << " bits/point" << endl;
          break;
        }
      }
      prev = current;
      iter++;
      TOTAL_ITERATIONS++;
    }
  }
  log.close();
}

void Mixture::CEM()
{
  /* prepare log file */
  string log_file = getLogFile();
  ofstream log(log_file.c_str());

  long double prev=0,current;
  int iter = 1,comp;
  printParameters(log,0,0);

  if (ESTIMATION == MML) {
    while (1) {
      comp = iter % K;
      cout << "comp updated: " << comp << endl;
      // Expectation (E-step)
      updateResponsibilityMatrix(comp);
      updateEffectiveSampleSize(comp);
      /*for (int i=0; i<K; i++) {
        if (sample_size[i] < 20) {
          current = computeMinimumMessageLength();
          goto stop;
        }
      }*/
      // Maximization (M-step)
      updateWeights(comp);
      updateComponents(comp);
      current = computeMinimumMessageLength();
      if (fabs(current) >= INFINITY) break;
      msglens.push_back(current);
      printParameters(log,iter,current);
      if (iter != 1) {
        assert(current > 0);
        // because EM has to consistently produce lower 
        // message lengths otherwise something wrong!
        // IMPORTANT: the below condition should not be 
        //          fabs(prev - current) <= 0.0001 * fabs(prev)
        // ... it's very hard to satisfy this condition and EM() goes into
        // ... an infinite loop!
        if (iter > 10 && (prev - current) <= IMPROVEMENT_RATE * prev) {
          stop:
          log << "\nSample size: " << N << endl;
          log << "encoding rate: " << current/N << " bits/point" << endl;
          break;
        }
      }
      prev = current;
      iter++;
      TOTAL_ITERATIONS++;
    } 
  } else if (ESTIMATION == ML) {  // ESTIMATION != MML
    while (1) {
      comp = iter % K;
      cout << "comp updated: " << comp << endl;
      // Expectation (E-step)
      updateResponsibilityMatrix(comp);
      updateEffectiveSampleSize(comp);
      // Maximization (M-step)
      updateWeights_ML(comp);
      updateComponents(comp);
      //current = negativeLogLikelihood(data);
      current = computeMinimumMessageLength();
      msglens.push_back(current);
      printParameters(log,iter,current);
      if (iter != 1) {
        //assert(current > 0);
        // because EM has to consistently produce lower 
        // -ve likelihood values otherwise something wrong!
        if (iter > 10 && (prev - current) <= IMPROVEMENT_RATE * prev) {
          current = computeMinimumMessageLength();
          log << "\nSample size: " << N << endl;
          log << "encoding rate (using ML): " << current/N << " bits/point" << endl;
          break;
        }
      }
      prev = current;
      iter++;
      TOTAL_ITERATIONS++;
    }
  }
  log.close();
}

/*!
 *  \brief This function returns the minimum message length of this mixture
 *  model.
 */
long double Mixture::getMinimumMessageLength()
{
  return minimum_msglen;
}

/*!
 *  \brief Returns the first part of the msg.
 */
long double Mixture::first_part()
{
  return part1;
}

/*!
 *  \brief Returns the second part of the msg.
 */
long double Mixture::second_part()
{
  return part2;
}

/*!
 *  \brief This function prints the parameters to a log file.
 *  \param os a reference to a ostream
 *  \param iter an integer
 *  \param msglen a long double
 */
void Mixture::printParameters(ostream &os, int iter, long double msglen)
{
  os << "Iteration #: " << iter << endl;
  for (int k=0; k<K; k++) {
    os << "\t" << fixed << setw(5) << "[" << k+1 << "]";
    os << "\t" << fixed << setw(10) << setprecision(3) << sample_size[k];
    os << "\t" << fixed << setw(10) << setprecision(5) << weights[k];
    os << "\t";
    components[k].printParameters(os);
  }
  os << "\t\t\tmsglen: " << msglen << " bits." << endl;
}

/*!
 *  \brief This function prints the parameters to a log file.
 *  \param os a reference to a ostream
 */
void Mixture::printParameters(ostream &os, int num_tabs)
{
  string tabs = "\t";
  if (num_tabs == 2) {
    tabs += "\t";
  }
  for (int k=0; k<K; k++) {
    os << tabs << "[";// << fixed << setw(5) << "[" << k+1 << "]";
    os << fixed << setw(2) << k+1;
    os << "]";
    os << "\t" << fixed << setw(10) << setprecision(3) << sample_size[k];
    os << "\t" << fixed << setw(10) << setprecision(5) << weights[k];
    os << "\t";
    components[k].printParameters(os);
  }
  os << tabs << "ID: " << id << endl;
  os << tabs << "MultivariateNormal encoding: " << minimum_msglen << " bits. "
     << "(" << minimum_msglen/N << " bits/point)" << endl << endl;
}

/*!
 *  \brief Outputs the mixture weights and component parameters to a file.
 */
void Mixture::printParameters(ostream &os)
{
  for (int k=0; k<K; k++) {
    os << "\t" << fixed << setw(10) << setprecision(5) << weights[k];
    os << "\t";
    components[k].printParameters(os);
  }
}

/*!
 *  \brief This function is used to read the mixture details to aid in
 *  visualization.
 *  \param file_name a reference to a string
 *  \param D an integer
 */
void Mixture::load(string &file_name, int D)
{
  sample_size.clear();
  weights.clear();
  components.clear();
  K = 0;
  ifstream file(file_name.c_str());
  string line;
  Vector numbers;
  Vector mean(D,0);
  Matrix cov(D,D);
  long double sum_weights = 0;
  while (getline(file,line)) {
    K++;
    boost::char_separator<char> sep("mucov,:()[] \t");
    boost::tokenizer<boost::char_separator<char> > tokens(line,sep);
    BOOST_FOREACH (const string& t, tokens) {
      istringstream iss(t);
      long double x;
      iss >> x;
      numbers.push_back(x);
    }
    weights.push_back(numbers[0]);
    sum_weights += numbers[0];
    for (int i=1; i<=D; i++) {
      mean[i-1] = numbers[i];
    }
    int k = D + 1;
    for (int i=0; i<D; i++) {
      for (int j=0; j<D; j++) {
        cov(i,j) = numbers[k++];
      }
    }
    MultivariateNormal mvnorm(mean,cov);
    components.push_back(mvnorm);
    numbers.clear();
  }
  file.close();
  for (int i=0; i<K; i++) {
    weights[i] /= sum_weights;
  }
}

/*!
 *  \brief This function is used to read the mixture details 
 *  corresponding to the given data.
 *  \param file_name a reference to a string
 *  \param D an integer
 *  \param d a reference to a std::vector<Vector>
 *  \param dw a reference to a Vector
 */
void Mixture::load(string &file_name, int D, std::vector<Vector> &d, Vector &dw)
{
  load(file_name,D);
  data = d;
  N = data.size();
  data_weights = dw;
  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);
  /*for (int i=0; i<K; i++) {
    responsibility.push_back(tmp);
  }*/
  updateResponsibilityMatrix();
  sample_size = Vector(K,0);
  updateEffectiveSampleSize();
  updateComponents();
  minimum_msglen = computeMinimumMessageLength();
}

/*!
 *  \brief This function is used to randomly choose a component.
 *  \return the component index
 */
int Mixture::randomComponent()
{
  long double random = uniform_random();
  //cout << random << endl;
  long double previous = 0;
  for (int i=0; i<weights.size(); i++) {
    if (random <= weights[i] + previous) {
      return i;
    }
    previous += weights[i];
  }
}

/*!
 *  \brief This function saves the data generated from a component to a file.
 *  \param index an integer
 *  \param data a reference to a std::vector<Vector>
 */
void Mixture::saveComponentData(int index, std::vector<Vector> &data)
{
  string data_file = "./visualize/comp";
  data_file += boost::lexical_cast<string>(index+1) + ".dat";
  //components[index].printParameters(cout);
  ofstream file(data_file.c_str());
  for (int j=0; j<data.size(); j++) {
    for (int k=0; k<data[0].size(); k++) {
      file << fixed << setw(10) << setprecision(3) << data[j][k];
    }
    file << endl;
  }
  file.close();
}

/*!
 *  \brief This function is used to randomly sample from the mixture
 *  distribution.
 *  \param num_samples an integer
 *  \param save_data a boolean variable
 *  \return the random sample
 */
std::vector<Vector> Mixture::generate(int num_samples, bool save_data) 
{
  sample_size = Vector(K,0);
  for (int i=0; i<num_samples; i++) {
    // randomly choose a component
    int k = randomComponent();
    sample_size[k]++;
  }
  ofstream fw("sample_size");
  for (int i=0; i<sample_size.size(); i++) {
    fw << sample_size[i] << endl;
  }
  fw.close();
  std::vector<Vector> sample;
  for (int i=0; i<K; i++) {
    std::vector<Vector> x = components[i].generate((int)sample_size[i]);
    if (save_data) {
      saveComponentData(i,x);
    }
    for (int j=0; j<x.size(); j++) {
      sample.push_back(x[j]);
    }
  }
  //return sample;
  for (size_t i = 0; i < num_samples; i++) {
    int idx1 = rand() % num_samples;
    int idx2 = rand() % num_samples;
    Vector tmp = sample[idx1];
    sample[idx1] = sample[idx2];
    sample[idx2] = tmp; 
  }
  return sample;
  // shuffle the sample
  /*std::vector<Vector> shuffled;
  std::vector<int> flags(num_samples,0);
  for (int i=0; i<num_samples; i++) {
    int index = rand() % num_samples;
    if (flags[index] == 0) {
      flags[index] = 1;
      shuffled.push_back(sample[index]);
    }
  }
  for (int i=0; i<num_samples; i++) {
    if (flags[i] == 0) {
      flags[i] = 1;
      shuffled.push_back(sample[i]);
    }
  }
  assert(shuffled.size() == num_samples);
  return shuffled;*/
}

/*!
 *  \brief This function splits a component into two.
 *  \return c an integer
 *  \param log a reference to a ostream
 *  \return the modified Mixture
 */
Mixture Mixture::split(int c, ostream &log)
{
  SPLITTING = 1;
  log << "\tSPLIT component " << c + 1 << " ... " << endl;

  int num_children = 2; 
  Mixture m(num_children,data,responsibility[c]);
  m.estimateParameters();
  log << "\t\tChildren:\n";
  m.printParameters(log,2); // print the child mixture

  // adjust weights
  Vector weights_c = m.getWeights();
  weights_c[0] *= weights[c];
  weights_c[1] *= weights[c];

  // adjust responsibility matrix
  std::vector<Vector> responsibility_c = m.getResponsibilityMatrix();
  for (int i=0; i<2; i++) {
    #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) 
    for (int j=0; j<N; j++) {
      responsibility_c[i][j] *= responsibility[c][j];
    }
  }

  // adjust effective sample size
  Vector sample_size_c(2,0);
  for (int i=0; i<2; i++) {
    long double sum = 0;
    #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) reduction(+:sum) 
    for (int j=0; j<N; j++) {
      sum += responsibility_c[i][j];
    }
    sample_size_c[i] = sum;
    if (sample_size_c[i] < MIN_N) {
      IGNORE_SPLIT = 1;
    }
  }

  // child components
  std::vector<MultivariateNormal> components_c = m.getComponents();

  // merge with the remaining components
  int K_m = K + 1;
  std::vector<Vector> responsibility_m(K_m);
  Vector weights_m(K_m,0),sample_size_m(K_m,0);
  std::vector<MultivariateNormal> components_m(K_m);
  int index = 0;
  for (int i=0; i<K; i++) {
    if (i != c) {
      weights_m[index] = weights[i];
      sample_size_m[index] = sample_size[i];
      responsibility_m[index] = responsibility[i];
      components_m[index] = components[i];
      index++;
    } else if (i == c) {
      for (int j=0; j<2; j++) {
        weights_m[index] = weights_c[j];
        sample_size_m[index] = sample_size_c[j];
        responsibility_m[index] = responsibility_c[j];
        components_m[index] = components_c[j];
        index++;
      }
    }
  }

  Vector data_weights_m(N,1);
  Mixture merged(K_m,components_m,weights_m,sample_size_m,responsibility_m,data,data_weights_m);
  log << "\t\tBefore adjustment ...\n";
  merged.computeMinimumMessageLength();
  merged.printParameters(log,2);
  if (IGNORE_SPLIT == 1) {
    log << "\t\tIGNORING SPLIT ... \n\n";
  } else {
    merged.EM();
    log << "\t\tAfter adjustment ...\n";
    merged.computeMinimumMessageLength();
    merged.printParameters(log,2);
    /*log << "Ik: " << IK << endl;
    log << "Iw: " << IW << endl;
    log << "Il: " << IL << endl;
    log << "It: " << sum_IT; print(log,IT,3); log << endl;*/
  }
  SPLITTING = 0;
  return merged;
}

/*!
 *  \brief This function deletes a component.
 *  \return c an integer
 *  \param log a reference to a ostream
 *  \return the modified Mixture
 */
Mixture Mixture::kill(int c, ostream &log)
{
  log << "\tKILL component " << c + 1 << " ... " << endl;

  int K_m = K - 1;
  // adjust weights
  Vector weights_m(K_m,0);
  long double residual_sum = 0;
  for (int i=0; i<K; i++) {
    if (i != c) {
      residual_sum += weights[i];
    }
  }
  long double wt;
  int index = 0;
  for (int i=0; i<K; i++) {
    if (i != c) {
      weights_m[index++] = weights[i] / residual_sum;
    }
  }

  // adjust responsibility matrix
  Vector residual_sums(N,0);
  for (int i=0; i<N; i++) {
    for (int j=0; j<K; j++) {
      if (j != c) {
        residual_sums[i] += responsibility[j][i];
      }
    }
    //if (residual_sums[i] < TOLERANCE) residual_sums[i] = TOLERANCE;
  }
  Vector resp(N,0);
  std::vector<Vector> responsibility_m(K_m,resp);
  index = 0;
  for (int i=0; i<K; i++) {
    if (i != c) {
      #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) //private(residual_sum) 
      for (int j=0; j<N; j++) {
        //residual_sum = 1 - responsibility[c][j];
        if (residual_sums[j] <= 0.06) {
          responsibility_m[index][j] = 1.0 / K_m;
        } else {
          responsibility_m[index][j] = responsibility[i][j] / residual_sums[j];
        }
        assert(responsibility_m[index][j] >= 0 && responsibility_m[index][j] <= 1);
      }
      index++;
    }
  }

  // adjust effective sample size
  Vector sample_size_m(K_m,0);
  for (int i=0; i<K-1; i++) {
    long double sum = 0;
    #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) reduction(+:sum) 
    for (int j=0; j<N; j++) {
      sum += responsibility_m[i][j];
    }
    sample_size_m[i] = sum;
  }

  // child components
  std::vector<MultivariateNormal> components_m(K_m);
  index = 0;
  for (int i=0; i<K; i++) {
    if (i != c) {
      components_m[index++] = components[i];
    }
  }

  log << "\t\tResidual:\n";
  Vector data_weights_m(N,1);
  Mixture modified(K_m,components_m,weights_m,sample_size_m,responsibility_m,data,data_weights_m);
  log << "\t\tBefore adjustment ...\n";
  modified.computeMinimumMessageLength();
  modified.printParameters(log,2);
  modified.EM();
  log << "\t\tAfter adjustment ...\n";
  //modified.computeMinimumMessageLength();
  modified.printParameters(log,2);
  return modified;
}

/*!
 *  \brief This function joins two components.
 *  \return c1 an integer
 *  \return c2 an integer
 *  \param log a reference to a ostream
 *  \return the modified Mixture
 */
Mixture Mixture::join(int c1, int c2, ostream &log)
{
  log << "\tJOIN components " << c1+1 << " and " << c2+1 << " ... " << endl;

  int K_m = K - 1;
  // adjust weights
  Vector weights_m(K_m,0);
  int index = 0;
  for (int i=0; i<K; i++) {
    if (i != c1 && i != c2) {
      weights_m[index++] = weights[i];
    }
  }
  weights_m[index] = weights[c1] + weights[c2];

  // adjust responsibility matrix
  std::vector<Vector> responsibility_m(K_m);
  index = 0;
  for (int i=0; i<K; i++) {
    if (i != c1 && i != c2) {
      responsibility_m[index++] = responsibility[i];
    }
  }
  Vector resp(N,0);
  #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) 
  for (int i=0; i<N; i++) {
    resp[i] = responsibility[c1][i] + responsibility[c2][i];
  }
  responsibility_m[index] = resp;

  // adjust effective sample size 
  Vector sample_size_m(K_m,0);
  index = 0;
  for (int i=0; i<K; i++) {
    if (i != c1 && i != c2) {
      sample_size_m[index++] = sample_size[i];
    }
  }
  sample_size_m[index] = sample_size[c1] + sample_size[c2];

  // child components
  std::vector<MultivariateNormal> components_m(K_m);
  index = 0;
  for (int i=0; i<K; i++) {
    if (i != c1 && i != c2) {
      components_m[index++] = components[i];
    }
  }
  Mixture joined(1,data,resp);
  joined.estimateParameters();
  log << "\t\tResultant join:\n";
  joined.printParameters(log,2); // print the joined pair mixture

  std::vector<MultivariateNormal> joined_comp = joined.getComponents();
  components_m[index++] = joined_comp[0];
  Vector data_weights_m(N,1);
  Mixture modified(K-1,components_m,weights_m,sample_size_m,responsibility_m,data,data_weights_m);
  log << "\t\tBefore adjustment ...\n";
  modified.computeMinimumMessageLength();
  modified.printParameters(log,2);
  modified.EM();
  log << "\t\tAfter adjustment ...\n";
  modified.printParameters(log,2);
  return modified;
}

/*!
 *  \brief This function generates data to visualize the 2D/3D heat maps.
 *  \param res a long double
 */
void Mixture::generateHeatmapData(long double res, int D)
{
  int N = 10000;
  sample_size = Vector(K,0);
  for (int i=0; i<N; i++) {
    int k = randomComponent();
    sample_size[k]++;
  }

  string mix_file = "./visualize/mixture_density.dat";
  ofstream mix(mix_file.c_str());

  long double comp_density,mix_density;
  for (int i=0; i<K; i++) {
    std::vector<Vector> x = components[i].generate(sample_size[i]);
    string comp_file = "./visualize/comp" + boost::lexical_cast<string>(i+1) 
                       + "_density.dat";
    ofstream comp(comp_file.c_str());
    for (int j=0; j<x.size(); j++) {
      mix_density = exp(log_probability(x[j]));
      comp_density = exp(components[i].log_density(x[j]));
      for (int k=0; k<x[j].size(); k++) {
        comp << fixed << setw(10) << setprecision(3) << x[j][k];
        mix << fixed << setw(10) << setprecision(3) << x[j][k];
      }
      comp << fixed << setw(10) << setprecision(4) << comp_density * 100 << endl;
      mix << fixed << setw(10) << setprecision(4) << mix_density * 100 << endl;
    }
    comp.close();
  }
  mix.close();

  if (D == 2) {
    string data_file = "./visualize/probability_density.dat";
    ofstream file(data_file.c_str());
    long double MIN = -10, MAX = 10;
    long double x1,x2; 
    Vector x(2,0);
    for (x1=MIN; x1<MAX; x1+=res) {
      for (x2=MIN; x2<MAX; x2+=res) {
        x[0] = x1; x[1] = x2;
        mix_density = exp(log_probability(x));
        file << fixed << setw(10) << setprecision(3) << x[0];
        file << fixed << setw(10) << setprecision(3) << x[1];
        file << fixed << setw(10) << setprecision(4) << mix_density * 100 << endl;
      } // x2
    } // x1
    file.close();
  } // if()
}

/*!
 *  \brief This function computes the nearest component to a given component.
 *  \param c an integer
 *  \return the index of the closest component
 */
int Mixture::getNearestComponent(int c)
{
  long double current,dist = LARGE_NUMBER;
  int nearest;

  for (int i=0; i<K; i++) {
    if (i != c) {
      current = components[c].computeKLDivergence(components[i]);
      if (current < dist) {
        dist = current;
        nearest = i;
      }
    }
  }
  return nearest;
}

