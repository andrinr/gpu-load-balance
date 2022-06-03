#include "pst.h"

class ServiceSetAdd : public mdl::BasicService {
    PST node_pst;
public:
    struct input {
        int idLower;
        int idUpper;
        input() = default;
        input(int idUpper) : idLower(0), idUpper(idUpper) {}
        input(int idLower, int idUpper) : idLower(idLower), idUpper(idUpper) {}
    };
    typedef void output;
    explicit ServiceSetAdd(PST node_pst)
        : BasicService(PST_SETADD, sizeof(input), "SetAdd"), node_pst(node_pst) {}
protected:
    virtual int operator()(int nIn, void *pIn, void *pOut) override;
    void SetAdd(PST pst,input *in);
};
