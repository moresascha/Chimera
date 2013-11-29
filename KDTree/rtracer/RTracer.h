#pragma once
#include <ChimeraAPI.h>
#include "../kdtree.cuh"

class IRTracer : public chimera::IGraphicSetting
{
public:

    IRTracer(LPCSTR name) : chimera::IGraphicSetting(name)
    {

    }

    virtual void VRender(void) = 0;

    virtual BOOL VOnRestore(UINT w, UINT h) = 0;

    chimera::CMShaderProgramDescription* VGetProgramDescription(VOID)
    {
        return NULL;
    }

    virtual ~IRTracer(void) { }
};

extern "C" IRTracer* createTracer(IKDTree* tree, int flags = NULL);

