all: fluid
#display

fluid: fluid.cu fluid.cpp fluid.h
	nvcc --expt-relaxed-constexpr -c fluid.cu -o kernel.o -lcudart -lcuda -lcublas
	#g++ -std=c++17 -O3 -lSDL2 -o fluid.o fluid.cpp
	nvcc -std=c++17 -O3 -x cu -o fluid.o fluid.cpp -lSDL2 -lcudart -lcuda -lcublas
	nvcc -std=c++17 kernel.o fluid.o -o main -lSDL2 -lcudart -lcuda -lcublas
	#g++ -std=c++17 -O3 -fopenmp -I/usr/local/include -L/usr/local/lib -lSDL2 -o fluid fluid.cpp
#-lSDL2_ttf -lomp
	# g++ -std=c++17 -o fluid fluid.cpp -lSDL2

#display: simDisplayer.cpp
#	g++ -std=c++17 -I/usr/local/include -L/usr/local/lib -I/usr/local/Cellar/sdl2/2.30.12/include -L/usr/local/Cellar/sdl2/2.30.12/lib -o display simDisplayer.cpp -lSDL2
