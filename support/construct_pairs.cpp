#include <iostream>
#include <cstdlib>
#include <vector>
#include <array>
#include <boost/program_options.hpp>

using namespace std;
using namespace boost::program_options;

struct Parameters
{
  int d;
};

struct TwoPairs 
{
  array<int,2> p1,p2;
};

struct Parameters parseCommandLineInput(int argc, char **argv)
{
  struct Parameters parameters;

  options_description desc("Allowed options");
  desc.add_options()
       ("d",value<int>(&parameters.d),"a +ve integer")
  ;
  variables_map vm;
  store(command_line_parser(argc,argv).options(desc).run(),vm);
  notify(vm);

  if (!vm.count("d")) {
    parameters.d = 3;
  }

  return parameters;
}

void print(array<int,2> &p)
{
  cout << "(" << p[0] << ", " << p[1] << ") ";
}

void print(vector<array<int,2> > &pairs)
{
  int num_pairs = pairs.size();
  cout << "pairs: " << num_pairs << "\t";

  cout << "< ";
  for (int i=0; i<num_pairs; i++) {
    print(pairs[i]);
  }
  cout << ">\n";
}

void print(vector<vector<TwoPairs> > &table)
{
  int dim = table.size();
  TwoPairs instance;

  cout << "dim: " << dim << endl;
  for (int i=0; i<dim; i++) {
    cout << "< ";
    for (int j=0; j<dim; j++) {
      instance = table[i][j];
      cout << "[ ";
      print(instance.p1);
      print(instance.p2);
      cout << "] ";
    } // j
    cout << ">\n";
  } // i
}

int main(int argc, char **argv)
{
  struct Parameters parameters = parseCommandLineInput(argc,argv);
  int d = parameters.d;

  cout << "D: " << d << endl;

  array<int,2> x;
  vector<array<int,2> > pairs;

  for (int i=0; i<d; i++) {
    x[0] = i;
    for (int j=i; j<d; j++) {
      x[1] = j;
      pairs.push_back(x);
    }
  }
  print(pairs);

  int dim = 0.5 * d * (d+1);  // == pairs.size()
  assert(dim == pairs.size());
  vector<TwoPairs> row(dim);
  vector<vector<TwoPairs> > table;
  TwoPairs instance;

  for (int i=0; i<dim; i++) {
    instance.p1 = pairs[i];
    for (int j=0; j<dim; j++) {
      instance.p2 = pairs[j];
      row[j] = instance;
    }
    table.push_back(row);
  }
  print(table);
}

