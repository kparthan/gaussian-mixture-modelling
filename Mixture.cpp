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
extern int MSGLEN_FAIL;

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
  D = components[0].getDimensionality();
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
  D = data[0].size();
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
  D = data[0].size();
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
    D = source.D;
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
    Ik = source.Ik;
    Iw = source.Iw;
    It = source.It;
    sum_It = source.sum_It;
    Il = source.Il;
    kd_term = source.kd_term;
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
  //cout << "Sample size: " << N << endl;

  // initialize responsibility matrix
  //srand(time(NULL));
  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);

  #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) 
  for (int i=0; i<N; i++) {
    int index = rand() % K;
    responsibility[index][i] = 1;
  }
  /*for (int i=0; i<N; i++) {
    if (K == 2) {
      if (i < 0.5 * N) {
        responsibility[0][i] = 1;
      } else {
        responsibility[1][i] = 1;
      }
    } else {
      int index = rand() % K;
      responsibility[index][i] = 1;
    }
  }*/
  /*for (int i=0; i<N; i++) {
    long double sum = 0;
    for (int j=0; j<K; j++) {
      long double random = uniform_random();
      responsibility[j][i] = random;
      sum += random;
    }
    for (int j=0; j<K; j++) {
      responsibility[j][i] /= sum;
    }
  }*/
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
  //N = data.size();
  //int D = data[0].size();
  cout << "Sample size: " << N << endl;

  long double init_weight = 1.0 / K;
  weights = Vector(K,init_weight);

  // determine covariance of the data
  Vector global_mean;
  Matrix global_cov;
  //Vector data_weights(N,1);
  computeMeanAndCovariance(data,data_weights,global_mean,global_cov);
  Vector diag_cov(D,0);
  for (int i=0; i<D; i++) {
    diag_cov[i] = 0.1 * global_cov(i,i);
  }
  int max_index = maximumIndex(diag_cov);
  long double max = diag_cov[max_index];
  Matrix cov = ZeroMatrix(D,D);
  for (int i=0; i<D; i++) {
    cov(i,i) = max;
  }
  //Matrix cov = 0.1 * global_cov;

  int trials=0,max_trials = 5;
  repeat:
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
  int success = updateResponsibilityMatrix();
  if (success == 0) {
    if (++trials <= max_trials) goto repeat;
    else {
      initialize(); //sleep(5);
      return;
    }
  }
  updateEffectiveSampleSize();
  updateWeights();
  success = updateComponents();
  if (success == 0) {
    if (++trials <= max_trials) goto repeat;
    else {
      initialize(); //sleep(5);
      return;
    }
  }
  success = updateResponsibilityMatrix();
  if (success == 0) {
    if (++trials <= max_trials) goto repeat;
    else {
      initialize(); //sleep(5);
      return;
    }
  }
  updateEffectiveSampleSize();
  for (int i=0; i<K; i++) {
    if (sample_size[i] < MIN_N) {
      cout << "... initialize2 failed ...\n";// sleep(5);
      initialize();
      return;
    }
  }
}

void Mixture::initialize3()
{
  int trials=0,max_trials = 5;
  repeat:
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
  // initialize memberships (hard)
  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);
  Vector distances(K,0);
  int nearest;
  for (int i=0; i<N; i++) {
    for (int j=0; j<K; j++) {
      distances[j] = data_weights[i] * computeEuclideanDistance(init_means[j],data[i]);
    } // for j()
    nearest = minimumIndex(distances);
    responsibility[nearest][i] = 1;
  } // for i()

  std::vector<Vector> means = init_means;
  int NUM_ITERATIONS = 10;
  for (int iter=0; iter<NUM_ITERATIONS; iter++) {
    // update means
    for (int i=0; i<K; i++) {
      long double neff = 0;
      means[i] = Vector(D,0);
      for (int j=0; j<N; j++) {
        if (responsibility[i][j] > 0.99) {
          neff += 1;
          for (int k=0; k<D; k++) {
            means[i][k] += data[j][k];
          } // for k
        } // if()
      } // for j
      for (int k=0; k<D; k++) {
        means[i][k] /= neff;
      } // for k
    } // for i 
    // update memberships
    for (int i=0; i<N; i++) {
      for (int j=0; j<K; j++) {
        distances[j] = data_weights[i] * computeEuclideanDistance(means[j],data[i]);
      }
      nearest = minimumIndex(distances);
      responsibility[nearest][i] = 1;
    }
  } // iter
  //cout << "init_means: ";
  for (int i=0; i<K; i++) {
    print(cout,init_means[i],3);
  }
  //cout << "\nk_means: ";
  for (int i=0; i<K; i++) {
    print(cout,means[i],3);
  } cout << endl;

  sample_size = Vector(K,0);
  updateEffectiveSampleSize();
  for (int i=0; i<K; i++) {
    if (sample_size[i] < MIN_N) {
      cout << "... initialize3 failed ...\n";// sleep(5);
      initialize();
      return;
    }
  }
  weights = Vector(K,0);
  updateWeights();

  // initialize parameters of each component
  Matrix cov;
  for (int i=0; i<K; i++) {
    cov = computeCovariance(data,responsibility[i],means[i]);
    MultivariateNormal mvnorm(means[i],cov);
    components.push_back(mvnorm);
  }
}

void Mixture::initialize4()
{
  assert(K == 2);
  Vector mean;
  Matrix cov;
  computeMeanAndCovariance(data,data_weights,mean,cov);

  // eigen decomposition of cov
  Vector eigen_values(D,0);
  Matrix eigen_vectors = IdentityMatrix(D,D);
  eigenDecomposition(cov,eigen_values,eigen_vectors);
  //cout << "eigen_values: "; print(cout,eigen_values,3); cout << endl;
  int max_eig = maximumIndex(eigen_values);
  Vector projection_axis(D,0);
  for (int i=0; i<D; i++) {
    projection_axis[i] = eigen_vectors(i,max_eig);
  }
  std::vector<Vector> init_means(K);
  init_means[0] = Vector(D,0);
  init_means[1] = Vector(D,0);
  long double add;
  for (int i=0; i<D; i++) {
    //add = sqrt(eigen_values[max_eig]) * projection_axis[i];
    add = eigen_values[max_eig] * projection_axis[i];
    init_means[0][i] = mean[i] + add; 
    init_means[1][i] = mean[i] - add;
  }

  /*std::vector<Vector> x_mu(N);
  Vector diff(D,0);
  for (int i=0; i<N; i++) {
    for (int j=0; j<D; j++) {
      diff[j] = data[i][j] - mean[j];
    }
    x_mu[i] = diff;
  }
  Vector projections(N,0);
  for (int i=0; i<N; i++) {
    projections[i] = data_weights[i] * computeDotProduct(x_mu[i],projection_axis);
  }
  int min_index = minimumIndex(projections);
  int max_index = maximumIndex(projections);
  cout << "max proj: " << projections[max_index] << endl;
  cout << "max proj: " << projections[max_index] << endl;
  for (int i=0; i<D; i++) {
    init_means[0][i] = mean[i] + projections[min_index] * projection_axis[i];
    init_means[1][i] = mean[i] + projections[max_index] * projection_axis[i];
  }*/

  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);
  Vector distances(K,0);
  int nearest;
  for (int i=0; i<N; i++) {
    for (int j=0; j<K; j++) {
      distances[j] = data_weights[i] * computeEuclideanDistance(init_means[j],data[i]);
    } // for j()
    nearest = minimumIndex(distances);
    responsibility[nearest][i] = 1;
  } // for i()

  sample_size = Vector(K,0);
  updateEffectiveSampleSize();
  for (int i=0; i<K; i++) {
    if (sample_size[i] < MIN_N) {
      cout << "... initialize4 failed ...\n"; sleep(5);
      initialize();
      return;
    }
  }
  weights = Vector(K,0);
  updateWeights();

  // initialize parameters of each component
  std::vector<Vector> responsibility2(K,tmp);
  for (int i=0; i<K; i++) {
    cov = computeCovariance(data,responsibility[i],init_means[i]);
    MultivariateNormal mvnorm(init_means[i],cov);
    components.push_back(mvnorm);
  }
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
int Mixture::updateComponents()
{
  Vector comp_data_wts(N,0);
  for (int i=0; i<K; i++) {
    #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) 
    for (int j=0; j<N; j++) {
      comp_data_wts[j] = responsibility[i][j] * data_weights[j];
    }
    components[i].estimateParameters(data,comp_data_wts);
    //components[i].updateParameters();
    Matrix cov = components[i].Covariance();
    for (int i=0; i<data[0].size(); i++) {
      if (cov(i,i) <= 0) {
        return 0;
      }
    }
  }
  return 1;
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
int Mixture::updateResponsibilityMatrix()
{
  //#pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) //private(j)
  for (int i=0; i<N; i++) {
    Vector log_densities(K,0);
    for (int j=0; j<K; j++) {
      log_densities[j] = components[j].log_density(data[i]);
      if (boost::math::isnan(log_densities[j]) ||
          fabs(log_densities[j]) >= INFINITY) return 0;
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
      //assert(!boost::math::isnan(responsibility[j][i]));
      if (boost::math::isnan(responsibility[j][i])) return 0;
    }
  }
  return 1;
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
    assert(!boost::math::isnan(log_densities[j]));
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
long double Mixture::computeNegativeLogLikelihood(std::vector<Vector> &sample)
{
  long double value = 0,log_density;
  #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) private(log_density) reduction(-:value)
  for (int i=0; i<sample.size(); i++) {
    log_density = log_probability(sample[i]);
    if(boost::math::isnan(log_density)) {
      writeToFile("resp",responsibility,3); 
    }
    assert(!boost::math::isnan(log_density));
    value -= log_density;
  }
  return value;
}

/*!
 *  \brief This function computes the minimum message length using the current
 *  model parameters.
 *  \return the minimum message length
 */
long double Mixture::computeMinimumMessageLength(int verbose /* default = 1 (print) */)
{
  MSGLEN_FAIL = 0;
  part1 = 0;
  part2 = 0;

  /****************** PART 1 *********************/

  // encode the number of components
  // assume uniform priors
  //long double Ik = log(MAX_COMPONENTS);
  Ik = K;
  //Ik = log(MAX_COMPONENTS) / log(2);
  //cout << "Ik: " << Ik << endl;

  // enocde the weights
  Iw = ((K-1)/2.0) * log(N);
  Iw -= boost::math::lgamma<long double>(K); // log(K-1)!
  for (int i=0; i<K; i++) {
    Iw -= 0.5 * log(weights[i]);
  }
  Iw /= log(2);
  //cout << "Iw: " << Iw << endl;
  //assert(Iw >= 0);

  // encode the parameters of the components
  It.clear();
  sum_It = 0;
  long double logp;
  for (int i=0; i<K; i++) {
    logp = components[i].computeLogParametersProbability(sample_size[i]);
    logp /= log(2);
    It.push_back(logp);
    sum_It += logp;
  }
  //cout << "It: " << sum_It << endl;
  /*if (It <= 0) { cout << It << endl;}
  fflush(stdout);
  assert(It > 0);*/

  // the constant term
  //int D = data[0].size();
  int num_free_params = (0.5 * D * (D+3) * K) + (K - 1);
  long double log_lattice_constant = logLatticeConstant(num_free_params);
  kd_term = 0.5 * num_free_params * log_lattice_constant;
  kd_term /= log(2);

  part1 = Ik + Iw + sum_It + kd_term;

  /****************** PART 2 *********************/

  // encode the likelihood of the sample
  long double Il_partial = computeNegativeLogLikelihood(data);
  Il = Il_partial - (D * N * log(AOM));
  Il /= log(2);
  //cout << "Il: " << Il << endl;
  //assert(Il > 0);
  if (Il < 0 || boost::math::isnan(Il)) {
    minimum_msglen = LARGE_NUMBER;
    MSGLEN_FAIL = 1;
    return minimum_msglen;
  }

  long double constant = 0.5 * num_free_params;
  constant /= log(2);
  part2 = Il + constant;

  minimum_msglen = part1 + part2;

  if (verbose == 1) {
    cout << "Ik: " << Ik << endl;
    cout << "Iw: " << Iw << endl;
    cout << "It: " << sum_It << endl;
    cout << "Il: " << Il << endl;
  }

  return minimum_msglen;
}

long double Mixture::computeApproximatedMessageLength()
{
  long double num_params = D * (D+3) * 0.5;
  long double npars_2 = num_params / 2;

  long double msglen = 0;

  long double Il = computeNegativeLogLikelihood(data);
  msglen += Il;

  long double Iw = 0;
  for (int i=0; i<K; i++) {
    Iw += log(weights[i]);
  }
  msglen += (npars_2 * Iw);

  long double tmp = (npars_2 + 0.5) * K * (1 - log(12) + log(N));
  //long double tmp = (npars_2 + 0.5) * K * log(N);
  msglen += tmp;

  return msglen / log(2);
}

void Mixture::printIndividualMsgLengths(ostream &log_file)
{
  log_file << "\t\tIk: " << Ik << endl;
  log_file << "\t\tIw: " << Iw << endl;
  log_file << "\t\tIt: " << sum_It << " "; print(log_file,It,3); log_file << endl;
  log_file << "\t\tlatt: " << kd_term << endl;
  log_file << "\t\tIl: " << Il << endl;
  log_file << "\t\tpart1 (Ik+Iw+It+latt): " << part1 << " + " 
           << "part2 (Il+d/(2*log(2))): " << part2 << " = "
           << part1 + part2 << " bits." << endl << endl;
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
  if (SPLITTING == 1) {
    initialize4();
  } else {
    //initialize();
    //initialize2();
    initialize3();
  }

  //initialize();

  //initialize2();

  //initialize3();

  //initialize4();

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

  long double impr_rate = 0.00001;
  /* EM loop */
  //if (ESTIMATION == MML) {
    while (1) {
      // Expectation (E-step)
      updateResponsibilityMatrix();
      updateEffectiveSampleSize();
      //if (SPLITTING == 1) {
        for (int i=0; i<K; i++) {
          if (sample_size[i] < MIN_N) {
            current = computeMinimumMessageLength();
            goto stop;
          }
        }
      //}
      // Maximization (M-step)
      updateWeights();
      updateComponents();
      current = computeMinimumMessageLength();
      if (fabs(current) >= INFINITY) break;
      msglens.push_back(current);
      printParameters(log,iter,current);
      if (iter != 1) {
        //assert(current > 0);
        // because EM has to consistently produce lower 
        // message lengths otherwise something wrong!
        // IMPORTANT: the below condition should not be 
        //          fabs(prev - current) <= 0.0001 * fabs(prev)
        // ... it's very hard to satisfy this condition and EM() goes into
        // ... an infinite loop!
        if ((iter > 3 && (prev - current) <= impr_rate * prev) ||
              (iter > 1 && current > prev) || current <= 0 || MSGLEN_FAIL == 1) {
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
  //}
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
      //current = computeNegativeLogLikelihood(data);
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
void Mixture::load(string &file_name, int dim)
{
  D = dim;
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
void Mixture::load(string &file_name, int dim, std::vector<Vector> &d, Vector &dw)
{
  D = dim;
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
  string data_file = "./visualize/sampled_data/comp";
  data_file += boost::lexical_cast<string>(index+1) + ".dat";
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
  /*ofstream fw("sample_size");
  for (int i=0; i<sample_size.size(); i++) {
    fw << sample_size[i] << endl;
  }
  fw.close();*/
  
  std::vector<std::vector<std::vector<long double> > > random_data;
  std::vector<Vector> sample;
  for (int i=0; i<K; i++) {
    std::vector<Vector> x = components[i].generate((int)sample_size[i]);
    random_data.push_back(x);
    for (int j=0; j<random_data[i].size(); j++) {
      sample.push_back(random_data[i][j]);
    }
  } // for i

  if (save_data) {
    writeToFile("random_sample.dat",sample);
    string comp_density_file;
    string mix_density_file = "./visualize/sampled_data/mixture_density.dat";
    ofstream mix(mix_density_file.c_str());
    long double comp_density,mix_density;
    for (int i=0; i<K; i++) {
      saveComponentData(i,random_data[i]);
      comp_density_file = "./visualize/sampled_data/comp" 
                          + boost::lexical_cast<string>(i+1) + "_density.dat";
      ofstream comp(comp_density_file.c_str());
      for (int j=0; j<random_data[i].size(); j++) {
        comp_density = exp(components[i].log_density(random_data[i][j]));
        mix_density = exp(log_probability(random_data[i][j]));
        for (int k=0; k<random_data[i][j].size(); k++) {
          comp << fixed << setw(10) << setprecision(3) << random_data[i][j][k];
          mix << fixed << setw(10) << setprecision(3) << random_data[i][j][k];
        } // k
        comp << "\t\t" << scientific << comp_density << endl;
        mix <<  "\t\t" << scientific << mix_density << endl;
      } // j
      comp.close();
    } // i
    mix.close();
  } // if()
  return sample;
  // shuffle the sample
  /*for (size_t i = 0; i < num_samples; i++) {
    int idx1 = rand() % num_samples;
    int idx2 = rand() % num_samples;
    Vector tmp = sample[idx1];
    sample[idx1] = sample[idx2];
    sample[idx2] = tmp; 
  }*/
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
    merged.printIndividualMsgLengths(log);
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
void Mixture::generateHeatmapData(int N, long double res, int D)
{
  generate(N,1);

  /*if (D == 2) {
    string data_file = "./visualize/sampled_data/probability_density.dat";
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
        file << fixed << setw(10) << setprecision(4) << mix_density << endl;
      } // x2
    } // x1
    file.close();
  } // if()
  */
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

/*!
 *  \brief Computes the KL-divergence between two mixtures
 */
long double Mixture::computeKLDivergence(Mixture &other)
{
  return computeKLDivergence(other,data);
}

// Monte Carlo
long double Mixture::computeKLDivergence(Mixture &other, std::vector<Vector> &sample)
{
  long double kldiv = 0,log_fx,log_gx;
  for (int i=0; i<sample.size(); i++) {
    log_fx = log_probability(sample[i]);
    log_gx = other.log_probability(sample[i]);
    kldiv += (log_fx - log_gx);
  }
  return kldiv/(log(2) * sample.size());
}

long double Mixture::computeKLDivergenceUpperBound(Mixture &other)
{
  Vector other_weights = other.getWeights();
  std::vector<MultivariateNormal> other_components = other.getComponents();
  int K2 = other_weights.size();

  // upper bounds
  long double log_cd1,log_cd2,log_cd,log_constant;
  Vector empty(K,0);

  std::vector<Vector> conflated_same(K,empty);
  for (int i=0; i<K; i++) {
    //log_cd1 = components[i].getLogNormalizationConstant();
    for (int j=0; j<K; j++) {
      //log_cd2 = components[j].getLogNormalizationConstant();
      MultivariateNormal conflated = components[i].conflate(components[j]);
      log_cd = conflated.getLogNormalizationConstant();
      //log_constant = log_cd1 + log_cd2 - log_cd;
      conflated_same[i][j] = exp(log_cd);
    } // j
  } // i

  long double kldiv;
  empty = Vector(K2,0);
  std::vector<Vector> exp_kldiv_diff(K,empty);
  for (int i=0; i<K; i++) {
    for (int j=0; j<K2; j++) {
      kldiv = components[i].computeKLDivergence(other_components[j]);
      exp_kldiv_diff[i][j] = exp(-kldiv);
    } // j
  } // i

  long double kl_upper_bound = 0;
  long double num,denom,diff;
  for (int i=0; i<K; i++) {
    num = 0;
    for (int j=0; j<K; j++) {
      num += (weights[i] * conflated_same[i][j]);
    }
    denom = 0;
    for (int j=0; j<K2; j++) {
      denom += (other_weights[j] * exp_kldiv_diff[i][j]);
    }
    diff = log(num) - log(denom);
    kl_upper_bound += (weights[i] * diff);
  }

  long double ent;
  for (int i=0; i<K; i++) {
    ent = components[i].entropy();
    kl_upper_bound += (weights[i] * ent);
  }

  return kl_upper_bound / log(2);
}

long double Mixture::computeKLDivergenceLowerBound(Mixture &other)
{
  Vector other_weights = other.getWeights();
  std::vector<MultivariateNormal> other_components = other.getComponents();
  int K2 = other_weights.size();

  // lower bounds
  long double kldiv;
  Vector empty(K,0);

  std::vector<Vector> exp_kldiv_same(K,empty);
  for (int i=0; i<K; i++) {
    for (int j=0; j<K; j++) {
      if (i != j) {
        kldiv = components[i].computeKLDivergence(components[j]);
        exp_kldiv_same[i][j] = exp(-kldiv);
      } else if (i == j) {
        exp_kldiv_same[i][j] = 1;
      }
    } // j
  } // i

  long double log_cd1,log_cd2,log_cd,log_constant;
  empty = Vector(K2,0);
  std::vector<Vector> conflated_diff(K,empty);
  for (int i=0; i<K; i++) {
    //log_cd1 = components[i].getLogNormalizationConstant();
    for (int j=0; j<K2; j++) {
      //log_cd2 = other_components[j].getLogNormalizationConstant();
      MultivariateNormal conflated = components[i].conflate(other_components[j]);
      log_cd = conflated.getLogNormalizationConstant();
      //log_constant = log_cd1 + log_cd2 - log_cd;
      conflated_diff[i][j] = exp(log_cd);
    } // j
  } // i

  long double kl_lower_bound = 0;
  long double num,denom,diff;
  for (int i=0; i<K; i++) {
    num = 0;
    for (int j=0; j<K; j++) {
      num += (weights[i] * exp_kldiv_same[i][j]);
    }
    denom = 0;
    for (int j=0; j<K2; j++) {
      denom += (other_weights[j] * conflated_diff[i][j]);
    }
    diff = log(num) - log(denom);
    kl_lower_bound += (weights[i] * diff);
  }

  long double ent;
  for (int i=0; i<K; i++) {
    ent = components[i].entropy();
    kl_lower_bound -= (weights[i] * ent);
  }

  //if (kl_lower_bound < 0) return 0;
  //else return kl_lower_bound / log(2);
  return kl_lower_bound / log(2);
}

long double Mixture::computeKLDivergenceAverageBound(Mixture &other)
{
  long double upper = computeKLDivergenceUpperBound(other);
  long double lower = computeKLDivergenceLowerBound(other);
  cout << "upper bound: " << upper << endl;
  cout << "lower bound: " << lower << endl;
  return 0.5 * (upper + lower);
}

