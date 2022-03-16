CC = mpiCC
RM=rm -f

SRCS=main.cc
OBJS=$(subst .cpp,.o,$(SRCS))

%.o: %.c $(DEPS)
	$(CC) -c -o $@ 

all: main

main: $(OBJS)
	$(CXX) -o main $(OBJS)

main.o: main.cpp

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) tool
