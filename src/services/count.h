#include "TraversePST.h"
#include "cell.h"
#include "../constants.h"

class ServiceCount : public TraverseCombinePST {
public:
    typedef struct Cell input; // Array of Cells
    typedef uint64_t output;   // Array of counts
    explicit ServiceCount(PST pst)
        : TraverseCombinePST(pst,PST_COUNT,MAX_CELLS*sizeof(input),MAX_CELLS*sizeof(output),"Count") {}
protected:
    virtual int Service(PST pst,void *vin,int nIn,void *vout,int nOut);
    virtual int Combine(void *vout,void *vout2,int nIn,int nOut1,int nOut2);
};
