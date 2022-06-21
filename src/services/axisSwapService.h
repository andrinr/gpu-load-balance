#include "TraversePST.h"
#include "cell.h"

class ServiceAxisSwap : public TraverseCombinePST {
public:
    static constexpr int max_cells = 8192;
    typedef struct Cell input; // Array of Cells
    typedef int output;   // Array of counts
    explicit ServiceAxisSwap(PST pst)
        : TraverseCombinePST(pst,PST_AXISSWAP,max_cells*sizeof(input),max_cells*sizeof(output),"Reshuffle") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
