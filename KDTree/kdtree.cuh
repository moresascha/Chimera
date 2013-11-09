#pragma once
#include <device_launch_parameters.h>
#include <cutil_inline.h>
#include <cutil_math.h>

struct SplitData
{
    int axis;
    float split;
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
    float max = fmaxf(dx, fmaxf(dy, dz));
    return max == dx ? 0 : max == dy ? 1 : 2;
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