CXX      = g++
FLAGS    = -fopenmp -O2 -march=native
OPTFLAGS = -fopenmp -O3 -march=native -ffast-math
LDFLAGS  = -lm

all: ques1 ques2 ques3

ques1: main.cpp correlate1.cpp
	$(CXX) $(FLAGS) -DQUES1 -o ques1 main.cpp correlate1.cpp $(LDFLAGS)

ques2: main.cpp correlate2.cpp
	$(CXX) $(FLAGS) -DQUES2 -o ques2 main.cpp correlate2.cpp $(LDFLAGS)

ques3: main.cpp correlate3.cpp
	$(CXX) $(OPTFLAGS) -DQUES3 -o ques3 main.cpp correlate3.cpp $(LDFLAGS)

clean:
	rm -f ques1 ques2 ques3

.PHONY: all clean
