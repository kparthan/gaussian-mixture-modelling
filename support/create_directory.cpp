#include <iostream>
#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <sstream>

using namespace std;

struct stat st;

int main(int argc, char **argv)
{
  if (stat("./some/", &st) == -1) {
      mkdir("./some/", 0700);
  }
  if (stat("./some2/", &st) == -1) {
      mkdir("./some2/", 0700);
  }
  string newf = "hello";
  if (stat(newf.c_str(), &st) == -1) {
      mkdir(newf.c_str(), 0700);
  }
}

