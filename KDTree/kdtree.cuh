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

__forceinline __device__ __host__ int getLongestAxis(float3 mini, float3 maxi) 
{
    float dx = maxi.x - mini.x;
    float dy = maxi.y - mini.y;
    float dz = maxi.z - mini.z;
    float m = max(dx, max(dy, dz));
    return m == dx ? 0 : m == dy ? 1 : 2;
}

struct float3min
{
    __device__ float3 operator()(float3 t0, float3 t1)
    {
        float3 r;
        r.x = nutty::binary::Min<float>()(t0.x, t1.x);
        r.y = nutty::binary::Min<float>()(t0.y, t1.y);
        r.z = nutty::binary::Min<float>()(t0.z, t1.z);
        return r;
    }
};

struct float3max
{
    __device__  float3 operator()(float3 t0, float3 t1)
    {
        float3 r;
        r.x = nutty::binary::Max<float>()(t0.x, t1.x);
        r.y = nutty::binary::Max<float>()(t0.y, t1.y);
        r.z = nutty::binary::Max<float>()(t0.z, t1.z);
        return r;
    }
};

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

struct IndexedSAH
{
    float sah;
    uint index;
};

struct Split_v1
{
    byte axis;
    uint primId;
    float split;
    IndexedSAH sah;
    uint below;
    uint above;
};

struct Split_v2
{
    byte* axis;
    uint* primId;
    float* split;
    IndexedSAH* sah;
    uint* below;
    uint* above;
};

struct Node_v1
{
    float split;
    uint contentStartIndex;
    uint contentCount;
    byte leaf;
    byte axis;
};

struct Node_v2
{
    float* split;
    uint* contentStartIndex;
    uint* contentCount;
    uint* below;
    byte* leaf;
    byte* axis;
};

enum EdgeType
{
    eStart,
    eEnd
};

struct _Edge
{
    float v;
    EdgeType type;
};

struct Indexed3DEdge_v1
{
    float3 t3;
    uint index;
};

struct Indexed3DEdge_v2
{
    _Edge t[3];
    uint index;
};

#define Indexed3DEdge Indexed3DEdge_v2

struct Edge_v1
{
    uint primId;
    Indexed3DEdge indexedEdge;
};

struct Edge_v2
{
    uint* primId;
    Indexed3DEdge* indexedEdge;
};

struct AABB_v1
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

#define Split Split_v2
#define Node Node_v2
#define Edge Edge_v2
#define AABB AABB_v1

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

__forceinline __device__ __host__ float getSplit(Edge e, uint id, byte axis)
{
    switch(axis)
    {
    case 0 : { return e.indexedEdge[id].t[0].v; }
    case 1 : { return e.indexedEdge[id].t[1].v; }
    case 2 : { return e.indexedEdge[id].t[2].v; }
    }
    return 0;
}

/*__forceinline __device__ void setEdgeSplit(Edge_v1* e, byte axis, float f)
{
    switch(axis)
    {
    case 0 : e->indexedEdge.t3.x = f; break;
    case 1 : e->indexedEdge.t3.y = f; break;
    case 2 : e->indexedEdge.t3.z = f; break;
    }
}*/

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
    virtual nutty::DeviceBuffer<Node>* GetNodes(void) = 0;
    virtual nutty::DeviceBuffer<AABB>* GetAABBs(void) = 0;
    virtual nutty::cuStream& GetDefaultStream(void) = 0;
};

struct SphereBBox
{
    __device__ AABB operator()(float3 pos)
    {
        AABB bbox;
        float bhe = 1;//sqrtf(1);
        bbox.min = pos - make_float3(bhe, bhe, bhe);
        bbox.max = pos + make_float3(bhe, bhe, bhe);

        return bbox;
    }
};