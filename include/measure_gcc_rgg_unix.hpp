#ifndef MEASURE_GCC_RGG_UNIX
#define MEASURE_GCC_RGG_UNIX

#include <cstdlib>
#include <string>
#include <unistd.h>
#include "measure_gcc_rgg.hpp"


void print_usage()
{
  std::string_view message = R"""(
NAME
  measure_gcc_rgg -- a program to measure giant connected component in function of \Delta\theta
SYNOPSIS
  measure_gcc_rgg [options] <filename with nodes' positions>
  )""";
  std::cout << message << std::endl;
}

void print_help()
{
  std::string_view help = R"""(
The following options are available:
  -d [DIMENSION]  Specify model's dimension (S^D).
  -s [STEP_THETA] Sampling resolution of theta
  -r              Use radius instead of theta
  )""";
  std::cout << help << std::endl;
}

bool parse_options(int argc , char *argv[], MeasureGCC &measureGCC)
{
  // Shows the options if no argument is given.
  if(argc == 1)
  {
    // print_usage();
    print_help();
    return false;
  }

  // Parsing options.
  int opt;
  while ((opt = getopt(argc,argv,"d:s:rn:i:e:")) != -1)
  {
    switch(opt)
    {
      case 'd':
        // Default ntimes=5
        measureGCC = MeasureGCC(std::stoi(optarg), 100, argv[argc - 1]);
        break;
      case 's':
        measureGCC.setThetaStep(std::stod(optarg));
        break;
      case 'r':
        measureGCC.setIsRadius(true);
        break;
      case 'n':
        measureGCC.setNtimes(std::stoi(optarg));
        break;
      case 'i':
        measureGCC.setInitTheta(std::stod(optarg));
        break;
      case 'e':
        measureGCC.setEndTheta(std::stod(optarg));
        break;
      default:
        print_usage();
        print_help();
        return false;
    }
  }
  return true;
}

#endif // MEASURE_GCC_RGG_UNIX