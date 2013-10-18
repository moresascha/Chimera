#pragma once
#include <device_launch_parameters.h>
#include <cutil_inline.h>
#include <cutil_math.h>

struct SplitData
{
    uint axis;
    float split;
};

struct AABB
{
    float4 min;
    float4 max;
};

struct SAHSplit
{
    float split;
    float v;
};

__device__ __host__ float getAxis(float4* vec, uint axis)
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

__device__ __host__ float getAxis(float3* vec, uint axis)
{
    float4 v = make_float4(vec->x, vec->y, vec->z, 0);
    return getAxis(&v, axis);
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

extern "C" void generate();