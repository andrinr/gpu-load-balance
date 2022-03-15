#include <stack>
#include "cells.h"
#include "constants.h"

#ifndef CELLS_H // include guard
#define CELLS_H

class Cells {

    public:
        int* cellInfo = new int[MAX_CELL_COUNT * 3];
        float* cellGeometry = new float[MAX_CELL_COUNT * DIMENSIONS * 2];

        void init() {
            // Init values
            cellGeometry[0] = -0.5;
            cellGeometry[1] = -0.5;
            cellGeometry[2] = -0.5;
            cellGeometry[3] = 0.5;
            cellGeometry[4] = 0.5;
            cellGeometry[5] = 0.5;

            cellInfo[0] = 0;
            cellInfo[1] = COUNT;

            stack.push(0);
        }

        void split(int axis, float position) {  
            int id = stack.top();
            int leftId = getLeftChild(id);
            int rightId = getRightChild(id);

            // Copy data
            for (int i = 0; i < DIMENSIONS; i++) {
                setCornerA(leftId, i,  getCornerA(id, + i));
                setCornerB(leftId, i,  getCornerB(id, + i));

                setCornerA(rightId, i,  getCornerA(id, + i));
                setCornerB(rightId, i,  getCornerB(id, + i));
            }


            setCornerB(leftId, axis, position);
            setBegin
            begin[childId] = begin[id];
            end[childId] = mid;
            stack.push(childId);

            childId += 1;

            // Right Child
            for (int i = 0; i < DIMENSIONS; i++) {
                cornerA[childId * DIMENSIONS + i] = cornerA[id * DIMENSIONS + i];
                cornerB[childId * DIMENSIONS + i] = cornerB[id * DIMENSIONS + i];
            }
            cornerA[childId * DIMENSIONS + axis] = split;
            begin[childId] = mid;
            end[childId] = end[id];
            stack.push(childId);

            childId += 1;

            stack.pop();
        }

        float getCornerA(int id, int axis) {
            return cellGeometry[id * DIMENSIONS + axis];
        }

        void setCornerA(int id, int axis, float value) {
            cellGeometry[id * DIMENSIONS + axis] = value;
        }

        float getCornerB(int id, int axis) {
            return cellGeometry[(id + 1) * DIMENSIONS + axis];
        }

        void setCornerB(int id, int axis, float value) {
            cellGeometry[(id + 1) * DIMENSIONS + axis] = value;
        }

        int getBegin(int id) {
            return cellInfo[id * 3];
        }

        void setBegin(int id, int value) {
            cellInfo[id * 3] = value;
        }

        int getEnd(int id) {
            return cellInfo[id * 3 + 1];
        }

        void setEnd(int id, int value) {
            cellInfo[id * 3 + 1] = value;
        }

        int getLeftChild(int id) {
            return cellInfo[id * 3 + 2];
        }

        int setLeftChild(int id, int value) {
            cellInfo[id * 3 + 2] + value;
        }

        int getRightChild(int id) {
            return getLeftChild(id) + 1;
        }

        int getId() {
            return stack.top();
        }

    private:
        std::stack<int> stack;

};

#endif // CELLS_H