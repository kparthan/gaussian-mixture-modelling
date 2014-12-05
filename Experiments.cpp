#include "Experiments.h"

extern int MIXTURE_SIMULATION;
extern int INFER_COMPONENTS;
extern int ENABLE_DATA_PARALLELISM;
extern int NUM_THREADS;
extern int ESTIMATION;
extern long double IMPROVEMENT_RATE;
extern int NUM_STABLE_COMPONENTS;
extern int TOTAL_ITERATIONS;
extern int TRUE_MIX,COMPARE1,COMPARE2;
struct stat st;

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

struct Parameters Experiments::setParameters(int N, int D)
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
  return parameters;
}

// ./experiments/infer_components/exp1/
// bivariate data, covariance = I
void Experiments::infer_components_exp1()
{
  iterations = 50;
  int N = 800;
  int D = 2;
  int precision = 1;
  //long double delta = 2.0;

  // iterations = 50 (in paper)
  struct Parameters parameters = setParameters(N,D);
    
  string results_folder = "./experiments/infer_components/exp1/";
  for (long double delta=1.8; delta<=2.6; delta+=0.1) {
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
    //generateExperimentalMixtures(original,delta,results_folder,N,precision);
    inferExperimentalMixtures(original,delta,results_folder,parameters,precision);
  }
}

void Experiments::infer_components_exp1_compare()
{
  iterations = 50;
  int D = 2;
  int precision = 1;
  //long double delta = 2.3;

  string folder1 = "./experiments/infer_components/exp1/";
  string folder2 = "./support/mixturecode2/exp1/";
  string true_mix,mix1_file,mix2_file,data_file;
  string delta_str,iter_str,output;
  struct Parameters parameters;
  std::pair<Vector,Vector> results;
  Vector msglens,kldivs;

  for (long double delta=1.8; delta<=2.6; delta+=0.1) {
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
    
    output = "./experiments/infer_components/exp1/comparisons/msglens_" + delta_str;
    ofstream out1(output.c_str());
    output = "./experiments/infer_components/exp1/comparisons/kldivs_" + delta_str;
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
    }
    out1.close();
    out2.close();
  }
}

void Experiments::infer_components_exp2()
{
  iterations = 50;
  int N = 800;
  int D = 10;
  int precision = 2;
  long double delta = 1.25;

  // iterations = 50 (in paper)
  struct Parameters parameters = setParameters(N,D);
    
  string results_folder = "./experiments/infer_components/exp2/";
  for (long double delta=1.10; delta<=1.60; delta+=0.05) {
    Vector mu1(D,0);
    Vector mu2(D,delta);
    Matrix C1 = IdentityMatrix(D,D);
    Matrix C2 = IdentityMatrix(D,D);

    MultivariateNormal mvnorm1(mu1,C1);
    MultivariateNormal mvnorm2(mu2,C2);

    Vector weights(2,0.5);
    std::vector<MultivariateNormal> components;
    components.push_back(mvnorm1);
    components.push_back(mvnorm2);
    Mixture original(2,components,weights);
    //generateExperimentalMixtures(original,delta,results_folder,N,precision);
    inferExperimentalMixtures(original,delta,results_folder,parameters,precision);
  }
}

void Experiments::infer_components_exp2_compare()
{
  iterations = 50;
  int D = 10;
  int precision = 2;
  long double delta = 1.60;

  string folder1 = "./experiments/infer_components/exp2/";
  string folder2 = "./support/mixturecode2/exp2/";
  string true_mix,mix1_file,mix2_file,data_file;
  string delta_str,iter_str,output;
  struct Parameters parameters;
  std::pair<Vector,Vector> results;
  Vector msglens,kldivs;

  for (long double delta=1.10; delta<=1.65; delta+=0.05) {
    Vector mu1(D,0);
    Vector mu2(D,delta);
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
    
    output = "./experiments/infer_components/exp2/comparisons/msglens_" + delta_str;
    ofstream out1(output.c_str());
    output = "./experiments/infer_components/exp2/comparisons/kldivs_" + delta_str;
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
    }
    out1.close();
    out2.close();
  }
}

void Experiments::generateExperimentalMixtures(
  Mixture &original, 
  long double delta,
  string &results_folder,
  int sample_size,
  int precision
) {
  std::ostringstream ss;
  ss << fixed << setprecision(precision);
  ss << delta;
  string delta_str = "delta_" + ss.str();
  
  string iter_str,data_file;
  std::vector<Vector> data;
  for (int iter=1; iter<=iterations; iter++) {
    iter_str = boost::lexical_cast<string>(iter);
    data_file = results_folder + "data/" + delta_str + "/mvnorm_iter_" + iter_str + ".dat";
    data = original.generate(sample_size,0);
    writeToFile(data_file,data);
  }
}

void Experiments::inferExperimentalMixtures(
  Mixture &original, 
  long double delta,
  string &results_folder,
  struct Parameters &parameters,
  int precision
) {
  std::ostringstream ss;
  ss << fixed << setprecision(precision);
  ss << delta;
  string delta_str = "delta_" + ss.str();

  string iter_str,infer_log,data_file,summary_log;
  summary_log = results_folder + "summary/" + delta_str;
  ofstream summary(summary_log.c_str());
  std::vector<Vector> data;
  int num_success = 0;
  for (int iter=1; iter<=iterations; iter++) {
    iter_str = boost::lexical_cast<string>(iter);
    infer_log = results_folder + "logs/" + delta_str + "/mvnorm_iter_" + iter_str + ".log";
    parameters.infer_log = infer_log;

    //data_file = results_folder + "data/" + delta_str + "/mvnorm_iter_" + iter_str + ".dat";
    data_file = "./support/mixturecode2/exp1/data/" + delta_str + "/"
                + "mvnorm_iter_" + iter_str + ".dat";
    cout << "data_file: " << data_file << endl;
    data = load_matrix(data_file,parameters.D);

    Vector data_weights(data.size(),1.0);
    Mixture m(parameters.start_from,data,data_weights);
    Mixture mixture = m;
    mixture.estimateParameters();
    ofstream log(parameters.infer_log.c_str());
    Mixture stable = inferComponents(mixture,data.size(),data[0].size(),log);
    log.close();
    string ans = results_folder + "mixtures/" + delta_str + "/mvnorm_iter_" + iter_str;
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
    if (stable.getNumberOfComponents() == original.getNumberOfComponents()) num_success++;
    summary << "\t\t" << iter << "\t\t" << stable.getNumberOfComponents() << "\t\t" 
            << TOTAL_ITERATIONS << endl;
  }
  summary << "\nsuccess rate: " << setprecision(2) 
          << num_success * 100 / (long double)(iterations) << " %\n";
  summary.close();
}

void Experiments::infer_components_increasing_sample_size_exp3()
{
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

  string folder = "./experiments/infer_components/exp3/";
  string N_str,iter_str,data_file,data_folder;
  std::vector<Vector> data;
  // generate data
/*
  //for (int N=600; N<=2000; N+=50) {
  for (int N=2050; N<=4000; N+=50) {
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
  long double avg_number;
  string avg_inference = folder + "avg_inference";
  ofstream avginfer(avg_inference.c_str(),ios::app);
  //for (int N=600; N<=2000; N+=50) {
  for (int N=2050; N<=4000; N+=50) {
    N_str = "N_" + boost::lexical_cast<string>(N);
    data_folder = folder + N_str + "/";
    summary_file = summary_folder + N_str;
    ofstream out(summary_file.c_str());
    avg_number = 0;
    parameters = setParameters(N,D);
    for (int iter=1; iter<=iterations; iter++) {
      iter_str = boost::lexical_cast<string>(iter);
      data_file = data_folder + "mvnorm_iter_" + iter_str + ".dat";
      data = load_matrix(data_file,D);
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
      avg_number += ans;
    } // for iter()
    out.close();
    avg_number /= iterations;
    avginfer << N << "\t\t" << avg_number << endl; 
  } // for N()
  avginfer.close();
}

void Experiments::infer_components_increasing_sample_size_exp4()
{
  int D = 10;
  int precision = 2;
  long double delta = 1.10;
  Vector mu1(D,0);
  Vector mu2(D,delta);
  Matrix C1 = IdentityMatrix(D,D);
  Matrix C2 = IdentityMatrix(D,D);
  MultivariateNormal mvnorm1(mu1,C1);
  MultivariateNormal mvnorm2(mu2,C2);
  Vector weights(2,0.5);
  std::vector<MultivariateNormal> components;
  components.push_back(mvnorm1);
  components.push_back(mvnorm2);
  Mixture original(2,components,weights);

  string folder = "./experiments/infer_components/exp4/";
  string N_str,iter_str,data_file,data_folder;
  std::vector<Vector> data;
  // generate data
/*
  for (int N=600; N<=2000; N+=50) {
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
  long double avg_number;
  string avg_inference = folder + "avg_inference";
  ofstream avginfer(avg_inference.c_str(),ios::app);
  for (int N=600; N<=2000; N+=50) {
    N_str = "N_" + boost::lexical_cast<string>(N);
    data_folder = folder + N_str + "/";
    summary_file = summary_folder + N_str;
    ofstream out(summary_file.c_str());
    avg_number = 0;
    parameters = setParameters(N,D);
    for (int iter=1; iter<=iterations; iter++) {
      iter_str = boost::lexical_cast<string>(iter);
      data_file = data_folder + "mvnorm_iter_" + iter_str + ".dat";
      data = load_matrix(data_file,D);
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
      avg_number += ans;
    } // for iter()
    out.close();
    avg_number /= iterations;
    avginfer << N << "\t\t" << avg_number << endl; 
  } // for N()
  avginfer.close();
}

