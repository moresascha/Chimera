#pragma once
#include "../../Nutty/Nutty/Nutty.h"
#include <device_launch_parameters.h>
#include <cutil_inline.h>

#include "../../Nutty/Nutty/Functions.h"
#include "../../Nutty/Nutty/DeviceBuffer.h"

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include <cutil_math.h>

#define DYNAMIC_PARALLELISM

#define GlobalId (blockDim.x * blockIdx.x + threadIdx.x)

#define EDGE_START ((byte)0)
#define EDGE_END   ((byte)1)

#define INVALID_ADDRESS ((uint)-1)
#define FLT_MAX 3.402823466e+38F
#define FLT_MAX_DIV2 (FLT_MAX/2.0f)
#define PI 3.14159265359
#define PI_MUL2 (2 * PI)

extern bool g_treeDebug;

__forceinline __device__ __host__ float getAxis(float4* vec, byte axis)
{
    switch(axis)
    {
    case 0 : return vec->x;
    case 1 : return vec->y;
    case 2 : return vec->z;
    case 3 : return vec->w;
    }
    return 0;
}

__forceinline __device__ __host__ void setAxis(float3* vec, byte axis, float v)
{
    switch(axis)
    {
    case 0 : vec->x = v; break;
    case 1 : vec->y = v; break;
    case 2 : vec->z = v; break;
    }
}

__forceinline __device__ __host__ float getAxis(float3* vec, byte axis)
{
    float4 v = make_float4(vec->x, vec->y, vec->z, 0);
    return getAxis(&v, axis);
}

__forceinline __device__ __host__ int getLongestAxis(float3* mini, float3* maxi) 
{
    float dx = maxi->x - mini->x;
    float dy = maxi->y - mini->y;
    float dz = maxi->z - mini->z;
    float m = max(dx, max(dy, dz));
    return m == dx ? 0 : m == dy ? 1 : 2;//return dx > dy && dx > dz ? 0 : (dy > dz ? 1 : 0);
}

__forceinline __device__ __host__ uint elemsBeforeLevel(byte l)
{
    return (1 << l) - 1;
}

__forceinline __device__ __host__ uint elemsOnLevel(byte l)
{
    return (1 << l);
}

__forceinline __device__ __host__ uint elemsBeforeNextLevel(byte l)
{
    return elemsBeforeLevel(l + 1);
}

struct BBox
{
    float3 min;
    float3 max;

    __device__ float get(byte axis, byte mm)
    {
        switch(mm)
        {
        case 0 : return getAxis(&min, axis);
        case 1 : return getAxis(&max, axis);
        }
        return 0;
    }

    __device__ float getX(byte mm)
    {
        return get(0, mm);
    }
    
    __device__ float getY(byte mm)
    {
        return get(1, mm);
    }

    __device__ float getZ(byte mm)
    {
        return get(2, mm);
    }
};

struct Primitive
{
    uint* primIndex;
    uint* nodeIndex;
    uint* prefixSum;
};

struct IndexedEdge
{
    uint index;
    float v;
};

struct Edge
{
    byte* type;
    IndexedEdge* indexedEdge;
    uint* nodeIndex;
    uint* primId;
    uint* prefixSum;
};

struct IndexedSplit
{
    float sah;
    uint index;
};

struct Split
{
    IndexedSplit* indexedSplit;
    byte* axis;
    uint* below;
    uint* above;
    float* v;
};

struct Node
{
    BBox* aabb;
    byte* isLeaf;
    float* split;
    byte* splitAxis;
    uint* contentCount;
    uint* contentStart;
    uint* content;
};

template<typename T>
__device__ __host__ void getSplit(T mmax, T mmin, float* split, byte* axis)
{
    float w = mmax.x - mmin.x;
    float h = mmax.y - mmin.y;
    float d = mmax.z - mmin.z;
    *axis = (w >= d && w >= h) ? 0 : (h >= w && h >= d) ? 1 : 2;
    switch(*axis)
    {
    case 0: 
        {
            *split = mmin.x + (mmax.x - mmin.x) * 0.5f;
        } break;
    case 1: 
        {
            *split = mmin.y + (mmax.y - mmin.y) * 0.5f;
            } break;
    case 2: 
        {
            *split = mmin.z + (mmax.z - mmin.z) * 0.5f;
        } break;
    }
}

class IKDTree
{
public:
    virtual void Generate(void) = 0;

    virtual void Init(void* mappedPtr, uint elements) = 0;

    virtual void SetDepth(uint d) = 0;

    virtual void Update(void) = 0;

    virtual uint GetCurrentDepth(void) = 0;

    virtual void GetContentCountStr(std::string& str) = 0;

    virtual void* GetData(void) = 0;

    virtual Node GetNodes(void) = 0;

    virtual nutty::DeviceBuffer<BBox>* GetAABBs(void) = 0;

    virtual nutty::cuStream& GetDefaultStream(void) = 0;
};

struct SphereBBox
{
    __device__ BBox operator()(float3 pos)
    {
        BBox bbox;
        float bhe = 1;//sqrtf(1);
        bbox.min = pos - make_float3(bhe, bhe, bhe);
        bbox.max = pos + make_float3(bhe, bhe, bhe);
        return bbox;
    }
};