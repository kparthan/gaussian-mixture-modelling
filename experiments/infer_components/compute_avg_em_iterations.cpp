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

typedef std::vector<double> Vector;

Vector process_summary_file(string &summary_file)
{
  Vector em_iterations(50,0);
  ifstream file(summary_file.c_str());
  string line;
  Vector numbers(3,0);
  int i,num_ems;
  for (int iter=0; iter<50; iter++) {
    getline(file,line);
    boost::char_separator<char> sep(" \t");
    boost::tokenizer<boost::char_separator<char> > tokens(line,sep);
    i = 0;
    BOOST_FOREACH(const string &t, tokens) {
      istringstream iss(t);
      double x;
      iss >> x;
      numbers[i++] = x;
    }
    num_ems = numbers[2];
    em_iterations[iter] = num_ems;
  } // for()
  file.close();
  /*cout << "summary: " << summary_file << endl;
  for (int i=0; i<em_iterations.size(); i++) {
    cout << em_iterations[i] << "\n";
  }*/
  return em_iterations;
}

double compute_average(Vector &values)
{
  double avg = 0;
  for (int i=0; i<values.size(); i++) {
    avg += values[i];
  }
  return avg / values.size();
}

int main(int argc, char **argv)
{
  /*string dir1 = "./exp1/summary/";
  string dir2 = "../../support/mixturecode2/exp1/summary/";
  string results = "./exp1/em_iterations";*/

  /*string dir1 = "./exp1a/summary/";
  string dir2 = "../../support/mixturecode2/exp1a/summary/";
  string results = "./exp1a/em_iterations";*/

  string dir1 = "./exp2/summary/";
  string dir2 = "../../support/mixturecode2/exp2/summary/";
  string results = "./exp2/em_iterations";

  string summary1,summary2;
  double delta,avg1,avg2;
  string delta_str;
  Vector em_iterations;
  ofstream out(results.c_str());

  //for (delta=1.8; delta<=2.65; delta+=0.1) {  // exp1,1a
  for (delta=1.10; delta<=1.61; delta+=0.05) {  // exp2
    ostringstream ssd;
    //ssd << fixed << setprecision(1);  // exp1,1a
    ssd << fixed << setprecision(2);    // exp2
    ssd << delta;
    delta_str = "delta_" + ssd.str();

    summary1 = dir1 + delta_str;
    em_iterations = process_summary_file(summary1);
    avg1 = compute_average(em_iterations);

    summary2 = dir2 + delta_str;
    em_iterations = process_summary_file(summary2);
    avg2 = compute_average(em_iterations);

    out << fixed << setw(10) << setprecision(2) << delta << "\t\t";
    out << setw(10) << setprecision(2) << avg1 << "\t\t";
    out << setw(10) << setprecision(2) << avg2 << "\t\n";
  } // for()
  out.close();
}

