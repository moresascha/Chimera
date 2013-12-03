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

enum DeviceBufferType
{
    eNodesContent,
    eNodesContentCount,
    eSplits,
    eNodeSplits,
    ePosSplits,
    eAxisAlignedBB,
    eBufferCount
};

struct Split
{
    int axis;
    uint primId;
    float split;
    float sah;
    uint below;
    uint above;
};

struct Node
{
    float split;
    uint contentStartIndex;
    uint contentCount;
    uint leaf;
    uint axis;
};

struct ContentPoint
{
    float3 pos;
    uint primId;
};

enum EdgeType
{
    eStart,
    eEnd
};

struct Edge
{
    EdgeType type;
    uint primId;
    float tx;
    float ty;
    float tz;

    __device__ __host__ float getSplit(byte axis)
    {
        switch(axis)
        {
        case 0 : { return tx; }
        case 1 : { return ty; }
        case 2 : { return tz; }
        }
        return 0;
    }

    __device__ void setSplit(byte axis, float f)
    {
        switch(axis)
        {
        case 0 : tx = f; break;
        case 1 : ty = f; break;
        case 2 : tz = f; break;
        }
    }

};

struct AABB
{
    float3 min;
    float3 max;
};

__forceinline __device__ __host__ float getAxis(float4* vec, uint axis)
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

__forceinline __device__ __host__ void setAxis(float3* vec, uint axis, float v)
{
    switch(axis)
    {
    case 0 : vec->x = v; break;
    case 1 : vec->y = v; break;
    case 2 : vec->z = v; break;
    }
}

__forceinline __device__ __host__ float getAxis(float3* vec, uint axis)
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

template<typename T>
__device__ __host__ void getSplit(T mmax, T mmin, float* split, uint* axis)
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
    virtual void* GetBuffer(DeviceBufferType id) = 0;
    virtual void* GetData(void) = 0;
    virtual nutty::DeviceBuffer<Node>* GetNodes(void) = 0;
};