#ifndef TASKS_H // include guard
#define TASKS_H

class Tasks {
public:

    static void operate(
            Orb& orb,
            MPI_Comm comm,
            blitz::Array<Cell, 1> cells,
            int nLeafCells);

    static void count(Orb& orb, MPI_Comm comm);

    static void reshuffle(Orb& orb, MPI_Comm comm);
};

#endif //TASKS_H