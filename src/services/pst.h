#ifndef PST_H
#define PST_H
#include "mdl.h"

class LocalData {
    //blitz::Array<> ...
};



//#include "bound.h"
class pstNode {
public:
    class pstNode *pstLower;
    LocalData *lcl;
    MDL mdl;
    int idSelf;
    int idUpper;
    int nLeaves;
    int nLower;
    int nUpper;
    //Bound bnd;
    pstNode(MDL mdl) : pstLower(nullptr), mdl(mdl),
        idSelf(mdlSelf(mdl)), nLeaves(1), nLower(0), nUpper(0) {}
    bool OffNode() const { return nLeaves > mdlCores(mdl); }
    bool OnNode()  const { return nLeaves <= mdlCores(mdl); }
    bool AmNode()  const { return nLeaves == mdlCores(mdl); }
    bool NotNode() const { return nLeaves != mdlCores(mdl); }
    bool AmCore()  const { return nLeaves == 1; }
    bool NotCore() const { return nLeaves > 1; }
};
typedef pstNode *PST;

enum pst_service {
    PST_SRV_STOP=0, /* service 0 is always STOP and handled by MDL */
    PST_SETADD,
    PST_COUNTLEFT,
};
#endif
