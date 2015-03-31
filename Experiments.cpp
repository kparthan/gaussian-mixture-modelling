include "Experiments.h"

extern int MIXTURE_SIMULATION;
extern int INFER_COMPONENTS;
extern int ENABLE_DATA_PARALLELISM;
extern int NUM_THREADS;
extern int ESTIMATION;
extern long double IMPROVEMENT_RATE;
extern int NUM_STABLE_COMPONENTS;
extern int TOTAL_ITERATIONS;
extern int TRUE_MIX,COMPARE1,COMPARE2;
extern int SPLIT_METHOD;
extern struct stat st;

Experiments::Experiments(int iterations) : iterations(iterations)
{
  iterations = 50;
}

struct Parameters Experiments::setParameters(int N, int D, int split_method)
{
  struct Parameters parameters;
  parameters.simulation = SET;
  parameters.load_mixture = UNSET;
  parameters.D = D;
  parameters.read_profiles = UNSET;
  parameters.mixture_model = SET;
  parameters.infer_num_components = SET;
  parameters.max_components = -1;
  parameters.continue_inference = UNSET;
  parameters.start_from = 1;
  parameters.heat_map = UNSET;
  parameters.sample_size = N;
  INFER_COMPONENTS = SET;
  MIXTURE_SIMULATION = SET;
  NUM_THREADS = 1;
  ENABLE_DATA_PARALLELISM = UNSET;
  ESTIMATION = MML;
  SPLIT_METHOD = split_method;
  return parameters;
}

// different split strategies
void Experiments::exp1()
{
  int N,split_method;

  N = 500;
  exp1_generate(N);

  for(int i=0; i<=6; i++) {
    //split_method = RANDOM_ASSIGNMENT_HARD;
    //split_method = RANDOM_ASSIGNMENT_SOFT;
    //split_method = MAX_VARIANCE_DETERMINISTIC;
    //split_method = MAX_VARIANCE_VARIABLE;
    //split_method = MIN_VARIANCE_DETERMINISTIC;
    //split_method = MIN_VARIANCE_VARIABLE;
    //split_method = KMEANS;
    //exp1_infer(N,split_method);
    exp1_infer(N,i);
  }

  exp1_infer_compare();
}

// ./experiments/search/exp1/
// bivariate data, covariance = I
void Experiments::exp1_generate(int N)
{
  int D = 2;
  int precision = 1;
  //long double delta = 2.0;

  string exp_folder = "./experiments/search/exp1/";
  check_and_create_directory(exp_folder);
  for (long double delta=1.8; delta<=2.65; delta+=0.1) {
    Vector mu1(D,0);
    Vector mu2(D,0); mu2[0] = delta;
    Matrix C1 = IdentityMatrix(D,D);
    Matrix C2 = IdentityMatrix(D,D);

    MultivariateNormal mvnorm1(mu1,C1);
    MultivariateNormal mvnorm2(mu2,C2);

    Vector weights(2,0.5);
    std::vector<MultivariateNormal> components;
    components.push_back(mvnorm1);
    components.push_back(mvnorm2);
    Mixture original(2,components,weights);
    generateExperimentalMixtures(original,delta,exp_folder,N,precision);
  }
}

void Experiments::exp1_infer(int N, int split_method)
{
  int D = 2;
  int precision = 1;
  //long double delta = 2.0;

  struct Parameters parameters = setParameters(N,D,split_method);
    
  string exp_folder = "./experiments/search/exp1/";
  for (long double delta=1.8; delta<=2.65; delta+=0.1) {
    Vector mu1(D,0);
    Vector mu2(D,0); mu2[0] = delta;
    Matrix C1 = IdentityMatrix(D,D);
    Matrix C2 = IdentityMatrix(D,D);

    MultivariateNormal mvnorm1(mu1,C1);
    MultivariateNormal mvnorm2(mu2,C2);

    Vector weights(2,0.5);
    std::vector<MultivariateNormal> components;
    components.push_back(mvnorm1);
    components.push_back(mvnorm2);
    Mixture original(2,components,weights);
    inferExperimentalMixtures(original,delta,exp_folder,parameters,precision);
  }
}

void Experiments::exp1_infer_compare()
{
  int D = 2;
  int precision = 1;
  //long double delta = 2.3;

  string exp_folder = "./experiments/search/exp1/";
  string true_mix,mix1_file,mix2_file,data_file;
  string delta_str,iter_str,output;
  struct Parameters parameters;
  std::pair<Vector,Vector> results;
  Vector msglens,kldivs;

  for (long double delta=1.8; delta<=2.65; delta+=0.1) {
    Vector mu1(D,0);
    Vector mu2(D,0); mu2[0] = delta;
    Matrix C1 = IdentityMatrix(D,D);
    Matrix C2 = IdentityMatrix(D,D);

    MultivariateNormal mvnorm1(mu1,C1);
    MultivariateNormal mvnorm2(mu2,C2);

    Vector weights(2,0.5);
    std::vector<MultivariateNormal> components;
    components.push_back(mvnorm1);
    components.push_back(mvnorm2);
    Mixture original(2,components,weights);

    std::ostringstream ss;
    ss << fixed << setprecision(precision);
    ss << delta;
    delta_str = "delta_" + ss.str();

    true_mix = "./simulation/mixture_" + delta_str;
    ofstream mix(true_mix.c_str());
    for (int i=0; i<2; i++) {
      mix << fixed << setw(10) << setprecision(5) << weights[i];
      mix << "\t";
      components[i].printParameters(mix);
    }
    mix.close();
    
    output = "./experiments/search/exp1/comparisons/msglens_" + delta_str;
    ofstream out1(output.c_str());
    output = "./experiments/search/exp1/comparisons/kldivs_" + delta_str;
    ofstream out2(output.c_str());
    for (int iter=1; iter<=iterations; iter++) {
      iter_str = boost::lexical_cast<string>(iter);
      //data_file = folder1 + "data/" + delta_str + "/mvnorm_iter_" + iter_str + ".dat";
      data_file = folder2 + "data/" + delta_str + "/mvnorm_iter_" + iter_str + ".dat";
      mix1_file = folder1 + "mixtures/" + delta_str + "/mvnorm_iter_" + iter_str;
      mix2_file = folder2 + "mixtures/" + delta_str + "/mvnorm_iter_" + iter_str;
      cout << "data: " << data_file << endl;
      cout << "mix1: " << mix1_file << endl;
      cout << "mix2: " << mix2_file << endl;

      parameters.comparison = SET;
      TRUE_MIX = SET; COMPARE1 = UNSET; COMPARE2 = SET;
      parameters.true_mixture = true_mix;
      parameters.other1_mixture = mix1_file;
      parameters.other2_mixture = mix2_file;
      parameters.read_profiles = SET;
      parameters.profile_file = data_file;
      parameters.D = D;
      results = compareMixtures(parameters);
      msglens = results.first; //print(cout,msglens,3);
      for (int i=0; i<msglens.size(); i++) {
        out1 << fixed << scientific << setprecision(6) << msglens[i] << "\t\t";
      }
      out1 << endl;
      kldivs = results.second; //print(cout,kldivs,3);
      for (int i=0; i<kldivs.size(); i++) {
        out2 << fixed << scientific << setprecision(6) << kldivs[i] << "\t\t";
      }
      out2 << endl;
    } // for() iter
    out1.close();
    out2.close();
  }
}

void Experiments::generateExperimentalMixtures(
  Mixture &original, 
  long double delta,
  string &exp_folder,
  int sample_size,
  int precision
) {
  std::ostringstream ss;
  ss << fixed << setprecision(precision);
  ss << delta;
  string delta_str = "delta_" + ss.str();

  string data_folder = exp_folder + "data/";
  check_and_create_directory(data_folder);
  data_folder += delta_str + "/";
  check_and_create_directory(data_folder);
  
  string iter_str,data_file;
  std::vector<Vector> data;
  for (int iter=1; iter<=iterations; iter++) {
    iter_str = boost::lexical_cast<string>(iter);
    data_file = data_folder + "mvnorm_iter_" + iter_str + ".dat";
    data = original.generate(sample_size,0);
    writeToFile(data_file,data);
  }
}

void Experiments::inferExperimentalMixtures(
  Mixture &original, 
  long double delta,
  string &exp_folder,
  struct Parameters &parameters,
  int precision
) {
  std::ostringstream ss;
  ss << fixed << setprecision(precision);
  ss << delta;
  string delta_str = "delta_" + ss.str();

  string data_folder = exp_folder + "data/" + delta_str + "/";

  string split_folder = "split_" + boost::lexical_cast<string>(split_method) + "/";
  string results_folder = exp_folder + split_folder;
  check_and_create_directory(results_folder);

  string logs_folder = results_folder + "logs/";
  check_and_create_directory(logs_folder);
  logs_folder += delta_str + "/";
  check_and_create_directory(logs_folder);

  string mixtures_folder = results_folder + "mixtures/";
  check_and_create_directory(mixtures_folder);
  mixtures_folder += delta_str + "/";
  check_and_create_directory(mixtures_folder);

  string summary_folder = results_folder + "summary/";
  check_and_create_directory(summary_folder);

  string iter_str,data_file,summary_log;
  summary_log = summary_folder + delta_str;
  ofstream summary(summary_log.c_str());
  std::vector<Vector> data;
  int num_success = 0;
  Vector inferred(iterations,0);
  long double avg_number,variance;
  for (int iter=1; iter<=iterations; iter++) {
    iter_str = boost::lexical_cast<string>(iter);
    parameters.infer_log = logs_folder + "mvnorm_iter_" + iter_str + ".log";

    data_file = data_folder + "mvnorm_iter_" + iter_str + ".dat";
    cout << "data_file: " << data_file << endl;
    data = load_data_table(data_file,parameters.D);

    Vector data_weights(data.size(),1.0);
    Mixture m(parameters.start_from,data,data_weights);
    Mixture mixture = m;
    mixture.estimateParameters();
    ofstream log(parameters.infer_log.c_str());
    Mixture stable = inferComponents(mixture,data.size(),data[0].size(),log);
    log.close();
    string ans = mixtures_folder + "mvnorm_iter_" + iter_str;
    ofstream out(ans.c_str());
    Vector weights = stable.getWeights();
    std::vector<MultivariateNormal> components = stable.getComponents();
    for (int k=0; k<components.size(); k++) {
      out << "\t" << fixed << setw(10) << setprecision(5) << weights[k];
      out << "\t";
      components[k].printParameters(out,1);
    }
    out.close();

    // update summary file
    inferred[iter-1] = stable.getNumberOfComponents();
    if (stable.getNumberOfComponents() == original.getNumberOfComponents()) num_success++;
    summary << "\t\t" << iter << "\t\t" << stable.getNumberOfComponents() << "\t\t" 
            << TOTAL_ITERATIONS << endl;
  }
  avg_number = computeMean(inferred);
  variance = computeVariance(inferred);
  summary << "\nsuccess rate: " //<< setprecision(2) 
          << num_success * 100 / (long double)(iterations) << " %\n";
  summary << "Avg: " << avg_number << endl;
  summary << "Variance: " << variance << endl;
  summary.close();
}

void Experiments::infer_components_increasing_sample_size_exp3()
{
  iterations = 50;
  int D = 2;
  int precision = 1;
  long double delta = 2.0;
  Vector mu1(D,0);
  Vector mu2(D,0); mu2[0] = delta;
  Matrix C1 = IdentityMatrix(D,D);
  Matrix C2 = IdentityMatrix(D,D);
  MultivariateNormal mvnorm1(mu1,C1);
  MultivariateNormal mvnorm2(mu2,C2);
  Vector weights(2,0.5);
  std::vector<MultivariateNormal> components;
  components.push_back(mvnorm1);
  components.push_back(mvnorm2);
  Mixture original(2,components,weights);

  string folder = "./experiments/search/exp3/delta_2.0/";
  string N_str,iter_str,data_file,data_folder;
  std::vector<Vector> data;
  // generate data
/*
  for (int N=600; N<=4000; N+=50) {
  //for (int N=2050; N<=4000; N+=50) {
    N_str = "N_" + boost::lexical_cast<string>(N);
    data_folder = folder + N_str + "/";
    if (stat(data_folder.c_str(), &st) == -1) {
        mkdir(data_folder.c_str(), 0700);
    }
    for (int iter=1; iter<=iterations; iter++) {
      data = original.generate(N,0);
      iter_str = boost::lexical_cast<string>(iter);
      data_file = data_folder + "mvnorm_iter_" + iter_str + ".dat";
      writeToFile(data_file,data);
    } // for iter()
  } // for N()
*/

  // infer mixture components
  struct Parameters parameters;
  string summary_folder = folder + "summary/";
  string summary_file;
  long double avg_number,variance;
  string avg_inference = folder + "avg_inference";
  ofstream avginfer(avg_inference.c_str(),ios::app);
  for (int N=600; N<=4000; N+=50) {
  //for (int N=2050; N<=4000; N+=50) {
    Vector inferred(iterations,0);
    N_str = "N_" + boost::lexical_cast<string>(N);
    data_folder = folder + N_str + "/";
    summary_file = summary_folder + N_str;
    ofstream out(summary_file.c_str());
    parameters = setParameters(N,D);
    for (int iter=1; iter<=iterations; iter++) {
      iter_str = boost::lexical_cast<string>(iter);
      data_file = data_folder + "mvnorm_iter_" + iter_str + ".dat";
      data = load_data_table(data_file,D);
      Vector data_weights(data.size(),1.0);
      Mixture m(parameters.start_from,data,data_weights);
      Mixture mixture = m;
      mixture.estimateParameters();
      parameters.infer_log = "dummy";
      ofstream log(parameters.infer_log.c_str());
      Mixture stable = inferComponents(mixture,data.size(),data[0].size(),log);
      log.close();
      int ans = stable.getNumberOfComponents();
      out << ans << endl;
      inferred[iter-1] = ans;
    } // for iter()
    out.close();
    avg_number = computeMean(inferred);
    variance = computeVariance(inferred);
    avginfer << N << "\t\t" << avg_number << "\t\t" << variance << endl; 
  } // for N()
  avginfer.close();

}

