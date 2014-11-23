#include "MixtureUnivariate.h"
#include "SupportUnivariate.h"

extern int MIXTURE_ID;
extern int MIXTURE_SIMULATION;
extern int INFER_COMPONENTS;
extern int ENABLE_DATA_PARALLELISM;
extern int NUM_THREADS;
extern long double IMPROVEMENT_RATE;
extern int ESTIMATION;
extern int TOTAL_ITERATIONS;
extern int SPLITTING;
extern int IGNORE_SPLIT;
extern long double MIN_N;
extern int MSGLEN_FAIL;

/*!
 *  \brief Null constructor module
 */
MixtureUnivariate::MixtureUnivariate()
{
  id = MIXTURE_ID++;
}

/*!
 *  \brief This is a constructor function which instantiates a MixtureUnivariate
 *  \param K an integer
 *  \param components a reference to a std::vector<Normal>
 *  \param weights a reference to a Vector 
 */
MixtureUnivariate::MixtureUnivariate(int K, std::vector<Normal> &components, Vector &weights):
                   K(K), components(components), weights(weights)
{
  assert(components.size() == K);
  assert(weights.size() == K);
  id = MIXTURE_ID++;
  D = 1;
  minimum_msglen = 0;
}

/*!
 *  \brief This is a constructor function.
 *  \param K an integer
 *  \param data a reference to a std::vector<Vector>
 *  \param data_weights a reference to a Vector
 */
MixtureUnivariate::MixtureUnivariate(int K, Vector &data, Vector &data_weights) : 
                   K(K), data(data), data_weights(data_weights)
{
  id = MIXTURE_ID++;
  N = data.size();
  D = 1; 
  assert(data_weights.size() == N);
  minimum_msglen = 0;
}

/*!
 *  \brief This is a constructor function.
 *  \param K an integer
 *  \param components a reference to a std::vector<Normal>
 *  \param weights a reference to a Vector
 *  \param sample_size a reference to a Vector
 *  \param responsibility a reference to a std::vector<Vector>
 *  \param data a reference to a std::vector<Vector>
 *  \param data_weights a reference to a Vector
 */
MixtureUnivariate::MixtureUnivariate(
  int K, 
  std::vector<Normal> &components, 
  Vector &weights,
  Vector &sample_size, 
  std::vector<Vector> &responsibility,
  Vector &data,
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
  D = 1;
  assert(data_weights.size() == N);
  minimum_msglen = 0;
}

/*!
 *  \brief This function assigns a source MixtureUnivariate distribution.
 *  \param source a reference to a MixtureUnivariate
 */
MixtureUnivariate MixtureUnivariate::operator=(const MixtureUnivariate &source)
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
 *  \brief This function checks whether the two MixtureUnivariate objects are the same.
 *  \param other a reference to a MixtureUnivariate
 *  \return whether they are the same object or not
 */
bool MixtureUnivariate::operator==(const MixtureUnivariate &other)
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
Vector MixtureUnivariate::getWeights()
{
  return weights;
}

/*!
 *  \brief This function returns the list of components.
 *  \return the components
 */
std::vector<Normal> MixtureUnivariate::getComponents()
{
  return components;
}

/*!
 *  \brief Gets the number of components
 */
int MixtureUnivariate::getNumberOfComponents()
{
  return components.size();
}

/*!
 *  \brief This function returns the responsibility matrix.
 */
std::vector<Vector> MixtureUnivariate::getResponsibilityMatrix()
{
  return responsibility;
}

/*!
 *  \brief This function returns the sample size of the mixture.
 *  \return the sample size
 */
Vector MixtureUnivariate::getSampleSize()
{
  return sample_size;
}

/*!
 *  \brief This function initializes the parameters of the model.
 */
void MixtureUnivariate::initialize()
{
  N = data.size();

  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);

  #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) 
  for (int i=0; i<N; i++) {
    int index = rand() % K;
    responsibility[index][i] = 1;
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
  components = std::vector<Normal>(K);
  updateComponents();
}

void MixtureUnivariate::initialize3()
{
  int trials=0,max_trials = 5;
  repeat:
  // choose K random means by choosing K random points
  Vector init_means(K,0);
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
  long double dist;
  for (int i=0; i<N; i++) {
    for (int j=0; j<K; j++) {
      dist = init_means[j] - data[i];
      distances[j] = data_weights[i] * dist * dist; 
    } // for j()
    nearest = minimumIndex(distances);
    responsibility[nearest][i] = 1;
  } // for i()

  Vector means = init_means;
  int NUM_ITERATIONS = 10;
  for (int iter=0; iter<NUM_ITERATIONS; iter++) {
    // update means
    for (int i=0; i<K; i++) {
      long double neff = 0;
      means[i] = 0; 
      for (int j=0; j<N; j++) {
        if (responsibility[i][j] > 0.99) {
          neff += 1;
          means[i] += data[j];
        } // if()
      } // for j
      means[i] /= neff;
    } // for i 
    // update memberships
    for (int i=0; i<N; i++) {
      for (int j=0; j<K; j++) {
        dist = init_means[j] - data[i];
        distances[j] = data_weights[i] * dist * dist; 
      }
      nearest = minimumIndex(distances);
      responsibility[nearest][i] = 1;
    }
  } // iter
  cout << "init_means: ";
  for (int i=0; i<K; i++) {
    cout << init_means[i] << "; ";
  }cout << endl;
  cout << "\nk_means: ";
  for (int i=0; i<K; i++) {
    cout << means[i] << "; ";
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
  long double sigma;
  for (int i=0; i<K; i++) {
    sigma = computeSigma(data,responsibility[i],means[i]);
    Normal norm(means[i],sigma);
    components.push_back(norm);
  }
}

void MixtureUnivariate::initialize4()
{
  assert(K == 2);
  long double mean,sigma;
  computeMeanAndSigma(data,data_weights,mean,sigma);

  Vector init_means(K,0);
  long double add = sigma;
  init_means[0] = mean + add; 
  init_means[1] = mean - add;

  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);
  Vector distances(K,0);
  int nearest;
  long double dist;
  for (int i=0; i<N; i++) {
    for (int j=0; j<K; j++) {
      dist = init_means[j] - data[i];
      distances[j] = data_weights[i] * dist * dist;
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
  for (int i=0; i<K; i++) {
    sigma = computeSigma(data,responsibility[i],init_means[i]);
    Normal norm(init_means[i],sigma);
    components.push_back(norm);
  }
}

/*!
 *  \brief This function updates the effective sample size of each component.
 */
void MixtureUnivariate::updateEffectiveSampleSize()
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

void MixtureUnivariate::updateEffectiveSampleSize(int index)
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
void MixtureUnivariate::updateWeights()
{
  long double normalization_constant = N + (K/2.0);
  for (int i=0; i<K; i++) {
    weights[i] = (sample_size[i] + 0.5) / normalization_constant;
  }
}

void MixtureUnivariate::updateWeights(int index)
{
  long double normalization_constant = N + (K/2.0);
  //for (int i=0; i<K; i++) {
    weights[index] = (sample_size[index] + 0.5) / normalization_constant;
  //}
}

void MixtureUnivariate::updateWeights_ML()
{
  long double normalization_constant = N;
  for (int i=0; i<K; i++) {
    weights[i] = sample_size[i] / normalization_constant;
  }
}

void MixtureUnivariate::updateWeights_ML(int index)
{
  long double normalization_constant = N;
  //for (int i=0; i<K; i++) {
    weights[index] = sample_size[index] / normalization_constant;
  //}
}

/*!
 *  \brief This function is used to update the components.
 */
int MixtureUnivariate::updateComponents()
{
  Vector comp_data_wts(N,0);
  for (int i=0; i<K; i++) {
    #pragma omp parallel for if(ENABLE_DATA_PARALLELISM) num_threads(NUM_THREADS) 
    for (int j=0; j<N; j++) {
      comp_data_wts[j] = responsibility[i][j] * data_weights[j];
    }
    components[i].estimateParameters(data,comp_data_wts);
  } // for()
  return 1;
}

void MixtureUnivariate::updateComponents(int index)
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
int MixtureUnivariate::updateResponsibilityMatrix()
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

void MixtureUnivariate::updateResponsibilityMatrix(int index)
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
void MixtureUnivariate::computeResponsibilityMatrix(Vector &sample, string &output_file)
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
long double MixtureUnivariate::log_probability(long double &x)
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
 *  \return the negative log likelihood (base e)
 */
long double MixtureUnivariate::computeNegativeLogLikelihood(Vector &sample)
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
long double MixtureUnivariate::computeMinimumMessageLength(int verbose /* default = 1 (print) */)
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
  int num_free_params = 3*K - 1;
  long double log_lattice_constant = logLatticeConstant(num_free_params);
  kd_term = 0.5 * num_free_params * log_lattice_constant;
  kd_term /= log(2);

  part1 = Ik + Iw + sum_It + kd_term;

  /****************** PART 2 *********************/

  // encode the likelihood of the sample
  long double Il_partial = computeNegativeLogLikelihood(data);
  Il = Il_partial - (N * log(AOM));
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

long double MixtureUnivariate::computeApproximatedMessageLength()
{
  long double num_params = 2 * K;
  long double npars_2 = num_params / 2;

  long double msglen = 0;

  long double Il = computeNegativeLogLikelihood(data);
  msglen += Il;

  long double Iw = 0;
  for (int i=0; i<K; i++) {
    Iw += log(weights[i]);
  }
  msglen += (npars_2 * Iw);

  long double tmp = (npars_2 + 0.5) * K * log(N);
  msglen += tmp;

  return msglen / log(2);
}

void MixtureUnivariate::printIndividualMsgLengths(ostream &log_file)
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
string MixtureUnivariate::getLogFile()
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
long double MixtureUnivariate::estimateParameters()
{
  if (SPLITTING == 1) {
    initialize4();
    //initialize();
  } else {
    initialize3();
  }

  EM();

  return minimum_msglen;
}

/*!
 *  \brief This function runs the EM method.
 */
void MixtureUnivariate::EM()
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

/*!
 *  \brief This function returns the minimum message length of this mixture
 *  model.
 */
long double MixtureUnivariate::getMinimumMessageLength()
{
  return minimum_msglen;
}

/*!
 *  \brief Returns the first part of the msg.
 */
long double MixtureUnivariate::first_part()
{
  return part1;
}

/*!
 *  \brief Returns the second part of the msg.
 */
long double MixtureUnivariate::second_part()
{
  return part2;
}

/*!
 *  \brief This function prints the parameters to a log file.
 *  \param os a reference to a ostream
 *  \param iter an integer
 *  \param msglen a long double
 */
void MixtureUnivariate::printParameters(ostream &os, int iter, long double msglen)
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
void MixtureUnivariate::printParameters(ostream &os, int num_tabs)
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
  os << tabs << "Normal encoding: " << minimum_msglen << " bits. "
     << "(" << minimum_msglen/N << " bits/point)" << endl << endl;
}

/*!
 *  \brief Outputs the mixture weights and component parameters to a file.
 */
void MixtureUnivariate::printParameters(ostream &os)
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
void MixtureUnivariate::load(string &file_name)
{
  D = 1;
  sample_size.clear();
  weights.clear();
  components.clear();
  K = 0;
  ifstream file(file_name.c_str());
  string line;
  Vector numbers;
  long double mean,sigma;
  long double sum_weights = 0;
  while (getline(file,line)) {
    K++;
    boost::char_separator<char> sep("musigacov,:()[] \t");
    boost::tokenizer<boost::char_separator<char> > tokens(line,sep);
    BOOST_FOREACH (const string& t, tokens) {
      istringstream iss(t);
      long double x;
      iss >> x;
      numbers.push_back(x);
    }
    weights.push_back(numbers[0]);
    sum_weights += numbers[0];
    mean = numbers[1];
    sigma = numbers[2];
    Normal norm(mean,sigma);
    components.push_back(norm);
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
void MixtureUnivariate::load(string &file_name, Vector &d, Vector &dw)
{
  D = 1;
  load(file_name);
  data = d;
  N = data.size();
  data_weights = dw;
  Vector tmp(N,0);
  responsibility = std::vector<Vector>(K,tmp);
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
int MixtureUnivariate::randomComponent()
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
void MixtureUnivariate::saveComponentData(int index, Vector &data)
{
  string data_file = "./visualize/sampled_data/comp";
  data_file += boost::lexical_cast<string>(index+1) + ".dat";
  ofstream file(data_file.c_str());
  for (int j=0; j<data.size(); j++) {
    file << fixed << setw(10) << setprecision(3) << data[j];
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
Vector MixtureUnivariate::generate(int num_samples, bool save_data) 
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
  
  std::vector<Vector> random_data;
  Vector sample;
  for (int i=0; i<K; i++) {
    Vector x = components[i].generate((int)sample_size[i]);
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
        comp << fixed << setw(10) << setprecision(3) << random_data[i][j];
        mix << fixed << setw(10) << setprecision(3) << random_data[i][j];
        comp << "\t\t" << scientific << comp_density << endl;
        mix <<  "\t\t" << scientific << mix_density << endl;
      } // j
      comp.close();
    } // i
    mix.close();
  } // if()
  return sample;
}

/*!
 *  \brief This function splits a component into two.
 *  \return c an integer
 *  \param log a reference to a ostream
 *  \return the modified MixtureUnivariate
 */
MixtureUnivariate MixtureUnivariate::split(int c, ostream &log)
{
  SPLITTING = 1;
  log << "\tSPLIT component " << c + 1 << " ... " << endl;

  int num_children = 2; 
  MixtureUnivariate m(num_children,data,responsibility[c]);
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
    /*if (sample_size_c[i] < MIN_N) {
      IGNORE_SPLIT = 1;
    }*/
  }

  // child components
  std::vector<Normal> components_c = m.getComponents();

  // merge with the remaining components
  int K_m = K + 1;
  std::vector<Vector> responsibility_m(K_m);
  Vector weights_m(K_m,0),sample_size_m(K_m,0);
  std::vector<Normal> components_m(K_m);
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
  MixtureUnivariate merged(K_m,components_m,weights_m,sample_size_m,responsibility_m,data,data_weights_m);
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
 *  \return the modified MixtureUnivariate
 */
MixtureUnivariate MixtureUnivariate::kill(int c, ostream &log)
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
  std::vector<Normal> components_m(K_m);
  index = 0;
  for (int i=0; i<K; i++) {
    if (i != c) {
      components_m[index++] = components[i];
    }
  }

  log << "\t\tResidual:\n";
  Vector data_weights_m(N,1);
  MixtureUnivariate modified(K_m,components_m,weights_m,sample_size_m,responsibility_m,data,data_weights_m);
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
 *  \return the modified MixtureUnivariate
 */
MixtureUnivariate MixtureUnivariate::join(int c1, int c2, ostream &log)
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
  std::vector<Normal> components_m(K_m);
  index = 0;
  for (int i=0; i<K; i++) {
    if (i != c1 && i != c2) {
      components_m[index++] = components[i];
    }
  }
  MixtureUnivariate joined(1,data,resp);
  joined.estimateParameters();
  log << "\t\tResultant join:\n";
  joined.printParameters(log,2); // print the joined pair mixture

  std::vector<Normal> joined_comp = joined.getComponents();
  components_m[index++] = joined_comp[0];
  Vector data_weights_m(N,1);
  MixtureUnivariate modified(K-1,components_m,weights_m,sample_size_m,responsibility_m,data,data_weights_m);
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
void MixtureUnivariate::generateHeatmapData(int N, long double res, int D)
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
int MixtureUnivariate::getNearestComponent(int c)
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
long double MixtureUnivariate::computeKLDivergence(MixtureUnivariate &other)
{
  return computeKLDivergence(other,data);
}

// Monte Carlo
long double MixtureUnivariate::computeKLDivergence(MixtureUnivariate &other, Vector &sample)
{
  long double kldiv = 0,log_fx,log_gx;
  for (int i=0; i<sample.size(); i++) {
    log_fx = log_probability(sample[i]);
    log_gx = other.log_probability(sample[i]);
    kldiv += (log_fx - log_gx);
  }
  return kldiv/(log(2) * sample.size());
}

long double MixtureUnivariate::computeKLDivergenceUpperBound(MixtureUnivariate &other)
{
  Vector other_weights = other.getWeights();
  std::vector<Normal> other_components = other.getComponents();
  int K2 = other_weights.size();

  // upper bounds
  long double log_cd1,log_cd2,log_cd,log_constant;
  Vector empty(K,0);

  std::vector<Vector> conflated_same(K,empty);
  for (int i=0; i<K; i++) {
    for (int j=0; j<K; j++) {
      Normal conflated = components[i].conflate(components[j]);
      log_cd = conflated.getLogNormalizationConstant();
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

  return kl_upper_bound;
}

long double MixtureUnivariate::computeKLDivergenceLowerBound(MixtureUnivariate &other)
{
  Vector other_weights = other.getWeights();
  std::vector<Normal> other_components = other.getComponents();
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
    for (int j=0; j<K2; j++) {
      Normal conflated = components[i].conflate(other_components[j]);
      log_cd = conflated.getLogNormalizationConstant();
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
  //else return kl_lower_bound;
  return kl_lower_bound;
}

long double MixtureUnivariate::computeKLDivergenceAverageBound(MixtureUnivariate &other)
{
  long double upper = computeKLDivergenceUpperBound(other) / log(2);
  long double lower = computeKLDivergenceLowerBound(other) / log(2);
  cout << "upper bound: " << upper << endl;
  cout << "lower bound: " << lower << endl;
  return 0.5 * (upper + lower);
}

