

class Cells {
    public:
        void init();
        void split();
        float getCornerA(int id, int axis);
        void setCornerA(int id, int axis);
        float getCornerB(int id, int axis);
        int getBegin(int id);
        int getStart(int id);
        int getLeftChild(int id);
        int getRigtChild(int id);
        int getId();

};