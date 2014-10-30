#CFLAGS=-std=c++0x -fopenmp -lmpfr $(shell pkg-config --cflags liblcb-experimental) 
#LDFLAGS=$(shell pkg-config --libs liblcb-experimental) -lboost_program_options -lboost_system -lboost_filesystem -fopenmp -lmpfr

# kernel-compatible version
#CFLAGS=-std=c++98 -c -O3 -I/home/parthan/external_libs/ -fopenmp 
#LDFLAGS=-static -lboost_program_options -lboost_filesystem -fopenmp -lm

#CFLAGS=-std=c++0x -c -O3 -fopenmp
CFLAGS=-std=c++0x -g -c -fopenmp
LDFLAGS=-lboost_program_options -lboost_system -lboost_filesystem -fopenmp -lm 

OBJECTS = main.o \
  Support.o \
  Structure.o \
  Normal.o \
  MultivariateNormal.o \
  Test.o \
  Experiments.o

all: main 

main: $(OBJECTS)
	g++ $(OBJECTS) -o $@ $(LDFLAGS) 

main.o: main.cpp Support.h Header.h
	g++ $(CFLAGS) $< -o $@

Support.o: Support.cpp Support.h Header.h UniformRandomNumberGenerator.h
	g++ $(CFLAGS) $< -o $@

Structure.o: Structure.cpp Structure.h Header.h
	g++ $(CFLAGS) $< -o $@

Normal.o: Normal.cpp Normal.h Header.h
	g++ $(CFLAGS) $< -o $@

MultivariateNormal.o: MultivariateNormal.cpp MultivariateNormal.h Header.h
	g++ $(CFLAGS) $< -o $@

#Mixture.o: Mixture.cpp Mixture.h Header.h 
#	g++ $(CFLAGS) $< -o $@

Test.o: Test.cpp Test.h Header.h MultivariateNormal.h
	g++ $(CFLAGS) $< -o $@

Experiments.o: Experiments.cpp Experiments.h Header.h
	g++ $(CFLAGS) $< -o $@

clean:
	rm -f *.o *~ main gmon.out 

