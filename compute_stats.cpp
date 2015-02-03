#include <iostream>
#include <fstream>
#include <sstream>
#include <vector> 
#include <cstdlib>
#include <iomanip>

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::program_options;
using namespace boost::filesystem;

typedef std::vector<long double> Vector;

#define NUM_METHODS 2

struct Parameters
{
  int n,d;
};

std::vector<Vector> load_table(string &file_name, int D)
{
  std::vector<Vector> sample;
  ifstream file(file_name.c_str());
  string line;
  Vector numbers(D,0);
  int i;
  while (getline(file,line)) {
    boost::char_separator<char> sep(" \t");
    boost::tokenizer<boost::char_separator<char> > tokens(line,sep);
    i = 0;
    BOOST_FOREACH(const string &t, tokens) {
      istringstream iss(t);
      long double x;
      iss >> x;
      numbers[i++] = x;
    }
    sample.push_back(numbers);
  }
  file.close();
  return sample;
}

std::vector<Vector> flip(std::vector<Vector> &table)
{
  int num_rows = table.size();
  Vector empty_vector(num_rows,0);
  int num_cols = table[0].size();
  std::vector<Vector> inverted_table(num_cols,empty_vector);
  for (int i=0; i<num_cols; i++) {
    for (int j=0; j<num_rows; j++) {
      inverted_table[i][j] = table[j][i];
    }
  }
  return inverted_table;
}

long double computeMean(Vector &list)
{
  long double sum = 0;
  for (int i=0; i<list.size(); i++) {
    sum += list[i];
  }
  return sum / (long double)list.size();
}

Vector computeMeans(std::vector<Vector> &table)
{
  std::vector<Vector> inverted_table = flip(table);
  int num_cols = table[0].size();
  Vector means(num_cols,0);
  for (int i=0; i<num_cols; i++) {
    means[i] = computeMean(inverted_table[i]);
  }
  return means;
}

Vector computeEstimateMeans(ostream &out, std::vector<Vector> &p_est_all)
{
  Vector means = computeMeans(p_est_all);
  int num_cols = p_est_all[0].size();
  for (int i=0; i<num_cols; i++) {
    out << scientific << means[i] << "\t";
  }
  out << endl;
  return means;
}

struct Parameters parseCommandLineInput(int argc, char **argv)
{
  struct Parameters parameters;

  options_description desc("Allowed options");
  desc.add_options()
       ("n",value<int>(&parameters.n),"sample size")
       ("d",value<int>(&parameters.d),"dimensionality")
  ;
  variables_map vm;
  store(command_line_parser(argc,argv).options(desc).run(),vm);
  notify(vm);

  return parameters;
}

void computeMeasures(
  int N,
  int D,
  std::vector<Vector> &negloglkhd,
  std::vector<Vector> &msglens,
  std::vector<Vector> &kldiv
) {
  string D_str = boost::lexical_cast<string>(D);
  string folder = "./experiments/D_" + D_str + "/";
  string means_file;

  means_file = folder + "means_negloglikelihood";
  ofstream logneg(means_file.c_str(),ios::app);
  logneg << fixed << setw(10) << N << "\t";
  Vector means_negloglkhd = computeEstimateMeans(logneg,negloglkhd);
  logneg.close();

  means_file = folder + "means_msglens";
  ofstream logmsg(means_file.c_str(),ios::app);
  logmsg << fixed << setw(10) << N << "\t";
  Vector means_msglens = computeEstimateMeans(logmsg,msglens);
  logmsg.close();

  means_file = folder + "means_kldiv";
  ofstream logkldiv(means_file.c_str(),ios::app);
  logkldiv << fixed << setw(10) << N << "\t";
  Vector means_kldiv = computeEstimateMeans(logkldiv,kldiv);
  logkldiv.close();
}

void computeWinsRatio(
  bool ignore_first, 
  const char *measure, 
  string input_file, 
  int N,
  string folder
) {
  ifstream input(input_file.c_str());
  // read values from the file
  string line;
  int start;
  if (ignore_first == 0) {
    start = 0;
  } else if (ignore_first == 1) {
    start = 1;
  }
  Vector numbers;
  std::vector<Vector> table;
  while (getline(input,line)) {
    boost::char_separator<char> sep(" \t");
    boost::tokenizer<boost::char_separator<char> > tokens(line,sep);
    BOOST_FOREACH (const string& t, tokens) {
      istringstream iss(t);
      long double x;
      iss >> x;
      numbers.push_back(x);
    }
    Vector values_to_compare;
    for (int i=start; i<numbers.size(); i++) {
      values_to_compare.push_back(numbers[i]);
    }
    table.push_back(values_to_compare);
    numbers.clear();
  }
  input.close();

  string wins_file = folder + "wins";
  ofstream out(wins_file.c_str(),ios::app);
  out << setw(10) << N << "\t";
  out << setw(20) << measure << "\t";
  std::vector<int> wins(NUM_METHODS,0);
  for (int i=0; i<table.size(); i++) {
    int winner = 0;
    long double min = table[i][0];
    for (int j=1; j<NUM_METHODS; j++) {
      if (table[i][j] <= min) {
        min = table[i][j];
        winner = j;
      }
    } // j loop ends ...
    wins[winner]++;
  } // i loop ends ...
  out << "[";
  for (int i=0; i<wins.size()-1; i++) {
    out << wins[i] << " : ";
  }
  out << wins[wins.size()-1] << "]\n";
  out.close();
}

int main(int argc, char **argv)
{
  struct Parameters parameters = parseCommandLineInput(argc,argv);

  string D_str = boost::lexical_cast<string>(parameters.d);
  string folder = "./experiments/D_" + D_str + "/";

  string size_str = boost::lexical_cast<string>(parameters.n);
  string negloglkhd_file = folder + "n_" + size_str + "_negloglikelihood";
  string kldiv_file = folder + "n_" + size_str + "_kldiv";
  string msglens_file = folder + "n_" + size_str + "_msglens";

  std::vector<Vector> negloglkhd = load_table(negloglkhd_file,NUM_METHODS+1);
  std::vector<Vector> kldiv = load_table(kldiv_file,NUM_METHODS);
  std::vector<Vector> msglens = load_table(msglens_file,NUM_METHODS+1);

  computeMeasures(parameters.n,parameters.d,negloglkhd,msglens,kldiv);

  computeWinsRatio(1,"negloglikhd",negloglkhd_file,parameters.n,folder);
  computeWinsRatio(1,"msglen",msglens_file,parameters.n,folder);
  computeWinsRatio(0,"kldiv",kldiv_file,parameters.n,folder);
}

