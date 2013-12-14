#pragma once
#include "../kdtree.cuh"
#include <cutil_math.h>
#include <ChimeraAPI.h>

class IRTracer;

#ifdef __cplusplus
extern "C" 
{
#endif

IRTracer* createTracer(IKDTree* tree, int flags = NULL);

#ifdef __cplusplus
}
#endif
    

class IRTracer : public chimera::IGraphicSetting
{
public:

    IRTracer(LPCSTR name) : chimera::IGraphicSetting(name)
    {

    }

    virtual void Compile(void) = 0;

    virtual void VRender(void) = 0;

    virtual int VOnRestore(uint w, uint h) = 0;

    virtual void ReleaseSharedResources(void) = 0;

    virtual void ToggleEnable(void) = 0;

    virtual double GetLastMillis(void) = 0;

    virtual uint GetLastRayCount(void) = 0;

    virtual uint GetLastShadowRayCount(void) = 0;

    chimera::CMShaderProgramDescription* VGetProgramDescription(VOID)
    {
        return NULL;
    }

    virtual ~IRTracer(void) { }
};

enum RayType
{
    eRayTypeShadow = 1,
    eRayTypeReflection = 2,
    eRayTypeRefraction = 3,
    eRayTypeInitial = 0
};

struct Ray
{
    uint2 screenCoord;
    float3 origin;
    float min;
    float3 dir;
    float max;
};

struct Sphere
{
    float3 pos;
    float radius;
    float4 color;
};