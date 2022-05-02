#ifndef TASKS_H // include guard
#define TASKS_H

class Tasks {
public:

    static void operate(Orb& orb, blitz::Array<Cell, 1> cells);

    static void work(Orb& orb, blitz::Array<Cell, 1> cells);
};

#endif //TASKS_H