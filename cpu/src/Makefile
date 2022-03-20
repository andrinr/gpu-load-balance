#HEADERS = Orb.h Cells.h constants.h
#OBJECTS = main.o Orb.o Cells.o

OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.h)

CC = mpiCC #g++

all: check program check

check:
	echo $(OBJECTS)
	echo $(HEADERS)

%.o: %.c $(HEADERS)
	$(CC) -c $< -o $@

program: $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

clean:
	-rm -f $(OBJECTS)
	-rm -f program

	