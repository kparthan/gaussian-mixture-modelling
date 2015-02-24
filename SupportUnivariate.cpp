#include "SupportUnivariate.h"

extern long double MIN_N;
extern int IGNORE_SPLIT;
extern int TRUE_MIX,COMPARE1,COMPARE2;
extern int TOTAL_ITERATIONS;

long double computeSigma(Vector &data, Vector &weights, long double &mean)
{
  int N = data.size();

  long double Neff = 0;
  long double diff,sqd_dev=0;
  for (int i=0; i<N; i++) {
    diff = data[i] - mean;
    sqd_dev += (weights[i] * diff * diff);
    Neff += weights[i];
  }
  if (Neff > 1) {
    sqd_dev /= (Neff - 1);
  } else {
    sqd_dev /= Neff;
  }

  return sqrt(sqd_dev);
}

void computeMeanAndSigma(
  Vector &data, 
  Vector &weights, 
  long double &mean,
  long double &sigma
) {
  int N = data.size();

  long double sum = 0,Neff=0;
  for (int i=0; i<N; i++) {
    sum += (weights[i] * data[i]);
    Neff += weights[i];
  }
  mean = sum / Neff;

  long double diff,sqd_dev=0;
  for (int i=0; i<N; i++) {
    diff = data[i] - mean;
    sqd_dev += (weights[i] * diff * diff);
  }
  if (Neff > 1) {
    sqd_dev /= (Neff - 1);
  } else {
    sqd_dev /= Neff;
  }
  sigma = sqrt(sqd_dev);
}

void writeToFile(const char *file_name, Vector &v)
{
  ofstream file(file_name);
  for (int i=0; i<v.size(); i++) {
    file << setw(15) << scientific << v[i] << endl;
  }
  file.close(); 
}

void writeToFile(string &file_name, Vector &v)
{
  ofstream file(file_name.c_str());
  for (int i=0; i<v.size(); i++) {
    file << setw(15) << scientific << v[i] << endl;
  }
  file.close(); 
}

std::vector<Normal> generateRandomComponentsUnivariate(int num_components)
{
  std::vector<Normal> components;
  long double mean,sigma;
  for (int i=0; i<num_components; i++) {
    mean = (uniform_random() - 0.5) * R1;
    sigma = uniform_random() * R2;
    Normal norm(mean,sigma);
    components.push_back(norm);
  }
  return components;
}

void computeEstimatorsUnivariate(struct Parameters &parameters, Vector &coordinates)
{
  if (parameters.mixture_model == UNSET) {  // no mixture modelling
    modelOneComponentUnivariate(parameters,coordinates);
  } else if (parameters.mixture_model == SET) { // mixture modelling
    modelMixtureUnivariate(parameters,coordinates);
  }
}

void modelOneComponentUnivariate(struct Parameters &parameters, Vector &data)
{
  cout << "Sample size: " << data.size() << endl;
  Normal norm;
  Vector weights(data.size(),1);
  struct EstimatesUnivariate estimates;
  norm.computeAllEstimators(data,estimates,1);
}

void modelMixtureUnivariate(struct Parameters &parameters, Vector &data)
{
  Vector data_weights(data.size(),1);
  // if the optimal number of components need to be determined
  if (parameters.infer_num_components == SET) {
    if (parameters.max_components == -1) {
      MixtureUnivariate mixture;
      if (parameters.continue_inference == UNSET) {
        MixtureUnivariate m(parameters.start_from,data,data_weights);
        mixture = m;
        mixture.estimateParameters();
      } else if (parameters.continue_inference == SET) {
        mixture.load(parameters.mixture_file,data,data_weights);
      } // continue_inference
      strategic_inference_univariate(parameters,mixture,data);
    }   
  } else if (parameters.infer_num_components == UNSET) {
    // for a given value of number of components
    // do the mixture modelling
    MixtureUnivariate mixture(parameters.fit_num_components,data,data_weights);
    mixture.estimateParameters();
  }
}

void simulateMixtureModelUnivariate(struct Parameters &parameters)
{
  Vector data;
  if (parameters.load_mixture == SET) {
    MixtureUnivariate original;
    original.load(parameters.mixture_file);
    bool save = 1;
    if (parameters.read_profiles == SET) {
      std::vector<Vector> data2;
      bool success = gatherData(parameters,data2);
      if (!success) {
        cout << "Error in reading data...\n";
        exit(1);
      }
      data = Vector(data2.size(),0);
      for (int i=0; i<data2.size(); i++) {
        data[i] = data2[i][0];
      }
    } else if (parameters.read_profiles == UNSET) {
      data = original.generate(parameters.sample_size,save);
    }
    /*if (parameters.heat_map == SET && (parameters.D == 2 || parameters.D == 3)) {
      original.generateHeatmapData(parameters.sample_size,parameters.res,parameters.D);
    }*/
  } else if (parameters.load_mixture == UNSET) {
    int k = parameters.simulated_components;
    Vector weights = generateFromSimplex(k);
    std::vector<Normal> components = generateRandomComponentsUnivariate(k);
    MixtureUnivariate original(k,components,weights);
    bool save = 1;
    data = original.generate(parameters.sample_size,save);
    // save the simulated mixture
    ofstream file("./simulation/simulated_mixture");
    for (int i=0; i<k; i++) {
      file << fixed << setw(10) << setprecision(5) << weights[i];
      file << "\t";
      components[i].printParameters(file);
    }
    file.close();
  }
  // model a mixture using the original data
  if (parameters.mixture_model == UNSET) {
    modelOneComponentUnivariate(parameters,data);
  } else if (parameters.mixture_model == SET) {
    modelMixtureUnivariate(parameters,data);
  }
}

void compareMixturesUnivariate(struct Parameters &parameters)
{
  MixtureUnivariate original,other1,other2;
  Vector data,dw;
  long double kldiv,msg,msg_approx;

  if (TRUE_MIX == SET) {
    original.load(parameters.true_mixture);
  }

  if (parameters.read_profiles == SET) {
    std::vector<Vector> data2;
    bool success = gatherData(parameters,data2);
    if (!success) {
      cout << "Error in reading data...\n";
      exit(1);
    }
    data = Vector(data2.size(),0);
    for (int i=0; i<data2.size(); i++) {
      data[i] = data2[i][0];
    }
  } else if (parameters.read_profiles == UNSET) {
    data = original.generate(parameters.sample_size,1);
  }
  dw = Vector(data.size(),1);

  if (TRUE_MIX == SET) {
    original.load(parameters.true_mixture,data,dw);
    msg = original.computeMinimumMessageLength(0);
    cout << "\n*** TRUE MIX ***\n";
    cout << "msg (true): " << msg << endl;
  }

  if (COMPARE1 == SET) {  // true mix is given
    cout << "\n*** OTHER1 MIX ***\n";
    other1.load(parameters.other1_mixture,data,dw);
    kldiv = original.computeKLDivergence(other1,data);
    cout << "kldiv (data): " << kldiv << endl;
    kldiv = original.computeKLDivergenceAverageBound(other1);
    cout << "kldiv (bound): " << kldiv << endl;
    msg = other1.computeMinimumMessageLength(0);
    cout << "msg (other1): " << msg << endl;
    msg_approx = other1.computeApproximatedMessageLength();
    cout << "msg_approx (other1): " << msg_approx << endl;
  }

  if (COMPARE2 == SET) {
    cout << "\n*** OTHER1 MIX ***\n";
    other1.load(parameters.other1_mixture,data,dw);
    msg = other1.computeMinimumMessageLength(0);
    cout << "msg (other1): " << msg << endl;
    msg_approx = other1.computeApproximatedMessageLength();
    cout << "msg_approx (other1): " << msg_approx << endl;
    if (TRUE_MIX == SET) {
      kldiv = original.computeKLDivergence(other1,data);
      cout << "kldiv (data): " << kldiv << endl;
      kldiv = original.computeKLDivergenceAverageBound(other1);
      cout << "kldiv (bound): " << kldiv << endl;
    }

    cout << "\n*** OTHER2 MIX ***\n";
    other2.load(parameters.other2_mixture,data,dw);
    msg = other2.computeMinimumMessageLength(0);
    cout << "msg (other2): " << msg << endl;
    msg_approx = other2.computeApproximatedMessageLength();
    cout << "msg_approx (other2): " << msg_approx << endl;
    if (TRUE_MIX == SET) {
      kldiv = original.computeKLDivergence(other2,data);
      cout << "kldiv (data): " << kldiv << endl;
      kldiv = original.computeKLDivergenceAverageBound(other2);
      cout << "kldiv (bound): " << kldiv << endl;
    }

    string mix1_density_file = "./visualize/sampled_data/inferred_mixture_1_density.dat";
    string mix2_density_file = "./visualize/sampled_data/inferred_mixture_2_density.dat";
    ofstream mix1(mix1_density_file.c_str());
    ofstream mix2(mix2_density_file.c_str());
    long double mix1_density,mix2_density;
    for (int j=0; j<data.size(); j++) {
      mix1_density = exp(other1.log_probability(data[j]));
      mix2_density = exp(other2.log_probability(data[j]));
      mix1 << fixed << setw(10) << setprecision(3) << data[j];
      mix2 << fixed << setw(10) << setprecision(3) << data[j];
      mix1 <<  "\t\t" << scientific << mix1_density << endl;
      mix2 <<  "\t\t" << scientific << mix2_density << endl;
    } // j
    mix1.close();
    mix2.close();
  } // if (compare2 == SET)
}

void strategic_inference_univariate(
  struct Parameters &parameters, 
  MixtureUnivariate &mixture, 
  Vector &data
) {
    ofstream log(parameters.infer_log.c_str());
    MixtureUnivariate stable = inferComponentsUnivariate(mixture,data.size(),log);
    cout << "# of components: " << stable.getNumberOfComponents() << endl;
    log.close();
    string ans = "./simulation/inferred_mixture_1";
    ofstream out(ans.c_str());
    Vector weights = stable.getWeights();
    std::vector<Normal> components = stable.getComponents();
    for (int k=0; k<components.size(); k++) {
      out << "\t" << fixed << setw(10) << setprecision(5) << weights[k];
      out << "\t";
      //components[k].printParameters(out);
      components[k].printParameters(out,1);
    }
    out.close();
}

MixtureUnivariate inferComponentsUnivariate(MixtureUnivariate &mixture, int N, ostream &log)
{
  int K,iter = 0;
  std::vector<Normal> components;
  MixtureUnivariate modified,improved,parent;
  Vector sample_size;
  std::vector<MixtureUnivariate> splits;

  MIN_N = 4;

  improved = mixture;

  while (1) {
    parent = improved;
    iter++;
    log << "Iteration #" << iter << endl;
    log << "Parent:\n";
    parent.printParameters(log,1);
    parent.printIndividualMsgLengths(log);
    components = parent.getComponents();
    sample_size = parent.getSampleSize();
    K = components.size();
    /*for (int i=0; i<K; i++) { // split() ...
      if (sample_size[i] > MIN_N) {
        IGNORE_SPLIT = 0;
        modified = parent.split(i,log);
        if (IGNORE_SPLIT == 0) {
          updateInference(modified,improved,N,log,SPLIT);
        }
      }
    }*/
    if (K >= 2) {  // kill() ...
      for (int i=0; i<K; i++) {
        modified = parent.kill(i,log);
        updateInference(modified,improved,N,log,KILL);
      } // killing each component
    } // if (K > 2) loop
    if (K > 1) {  // join() ...
      for (int i=0; i<K; i++) {
        int j = parent.getNearestComponent(i); // closest component
        modified = parent.join(i,j,log);
        updateInference(modified,improved,N,log,JOIN);
      } // join() ing nearest components
    } // if (K > 1) loop
    if (improved == parent) {
      splits.clear();
      for (int i=0; i<K; i++) { // split() ...
        if (sample_size[i] > MIN_N) {
          IGNORE_SPLIT = 0;
          modified = parent.split(i,log);
          splits.push_back(modified);
          if (IGNORE_SPLIT == 0) {
            updateInference(modified,improved,N,log,SPLIT);
          }
        }
      } // for()
    }
    if (improved == parent) goto finish;
  } // if (improved == parent || iter%2 == 0) loop

  finish:
  cout << "TOTAL_ITERATIONS: " << TOTAL_ITERATIONS << endl;
  return parent;
}

/*!
 *  \brief Updates the inference
 *  \param modified a reference to a MixtureUnivariate
 *  \param current a reference to a MixtureUnivariate
 *  \param log a reference to a ostream
 *  \param operation an integer
 */
void updateInference(MixtureUnivariate &modified, MixtureUnivariate &current, int N, ostream &log, int operation)
{
  long double modified_msglen = modified.getMinimumMessageLength();
  long double current_msglen = current.getMinimumMessageLength();

  long double dI = current_msglen - modified_msglen;
  long double dI_n = dI / N;
  long double improvement_rate = (current_msglen - modified_msglen) / current_msglen;

  if (operation == KILL || operation == JOIN || operation == SPLIT) {
    if (improvement_rate >= 0) {
      log << "\t ... IMPROVEMENT ... (+ " << fixed << setprecision(3) 
          << 100 * improvement_rate << " %) ";
      log << "\t\t[ACCEPT]\n\n";
      current = modified;
    } else {
      log << "\t ... NO IMPROVEMENT\t\t\t[REJECT]\n\n";
    }
  } 
}

