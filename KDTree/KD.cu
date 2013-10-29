#pragma once
#include "kdtree.cuh"

#define GlobalId (blockDim.x * blockIdx.x + threadIdx.x)

#define INVALID_ADDRESS ((uint)-1)
#define FLT_MAX 3.402823466e+38F


//todo: template it
extern "C" __global__ void scanKD(uint* content, uint* contentCount, uint neutralItem, uint memoryOffset, uint depth)
{
    extern __shared__ uint shrdMem[];

    uint thid = threadIdx.x;
    uint grpId = blockIdx.x;
    uint N = 2 * blockDim.x;
    uint offset = 1;
    
    uint i0 = content[memoryOffset + 2*thid + grpId * N];
    uint i1 = content[memoryOffset + 2*thid + 1 + grpId * N];

    if(neutralItem == i0)
    {
        shrdMem[2 * thid] = 0;
    }
    else
    {
        shrdMem[2 * thid] = 1;
    }

    if(neutralItem == i1)
    {
        shrdMem[2 * thid + 1] = 0;
    }
    else
    {
        shrdMem[2 * thid + 1] = 1;
    }

    uint last = shrdMem[2*thid+1];

    for(int d = N>>1; d > 0; d >>= 1)
    {   
        __syncthreads();
        if (thid < d)  
        {
            int ai = offset*(2*thid+1)-1;  
            int bi = offset*(2*thid+2)-1;
            shrdMem[bi] += shrdMem[ai];
        }
        offset *= 2;
    }
    
    if (thid == 0) { shrdMem[N - 1] = 0; }
        
    for(int d = 1; d < N; d *= 2)
    {  
        offset >>= 1;  
        __syncthreads();
        if (thid < d)                       
        {  
            int ai = offset*(2*thid+1)-1;  
            int bi = offset*(2*thid+2)-1;  
            int t = shrdMem[ai];  
            shrdMem[ai] = shrdMem[bi];  
            shrdMem[bi] += t;   
        } 
    }
    
    __syncthreads();

    if(neutralItem != i0)
    {
        content[memoryOffset + grpId * N + shrdMem[2*thid]] = i0;
    }

    if(neutralItem != i1)
    {
        content[memoryOffset + grpId * N + shrdMem[2*thid+1]] = i1;
    }
    
    if(thid == (blockDim.x-1))
    {
        contentCount[((1 << (depth))-1) + grpId] = shrdMem[2*thid+1] + last;
    }
}

__device__ SplitData* getParentSplit(SplitData* splitData, uint* nodesContent, uint* address, uint depth)
{
    return 0;
}

extern "C" __global__ void spreadContent(uint* nodesContent, uint* nodesContentCount, SplitData* splitData, float3* data, uint count, uint offset, uint depth)
{
    uint id = GlobalId;
    uint address = count * depth;
    uint contentCountStartAdd = (1 << depth) - 1;

    uint sa = 0;
    uint last = 1;
    uint m = 0;
    
    for(uint i = 0; i < (1 << depth); ++i)
    {
        last = nodesContentCount[contentCountStartAdd];
        sa += last;
        if(id < sa)
        {
            break;
        }
        m++;
        contentCountStartAdd++;
        address += count;
    }

    SplitData sd = splitData[contentCountStartAdd];
    
    address += (id % last);

    uint add = nodesContent[address];

    if(add == INVALID_ADDRESS)
    {
        return;
    }

    float3 d = data[add];

    uint memoryOffset = count * ((1 << (depth+1)) - 1) + 2 * m * count;

    if(sd.split < getAxis(&d, sd.axis))
    {
        nodesContent[memoryOffset + id] = add;
    }
    else
    {
        nodesContent[memoryOffset + count + id] = add;
    }
}

float3 __device__ getBHE(AABB* aabb)
{
    return make_float3(aabb->max.x - aabb->min.x, aabb->max.y - aabb->min.y, aabb->max.z - aabb->min.z);
}

float __device__ getArea(AABB* aabb)
{
    float3 bhe = getBHE(aabb);
    return 2 * bhe.x * bhe.y + 2 * bhe.x * bhe.z + 2 * bhe.y * bhe.z;
}

float __device__ getSAH(AABB* node, int axis, float split, int primAbove, int primBelow, float traversalCost = 0, float isectCost = 1)
{
    float3 bhe = getBHE(node);
    float invTotalSA = 1.0f / getArea(node);
    int otherAxis0 = (axis+1) % 3;
    int otherAxis1 = (axis+2) % 3;
    float belowSA = 
            2 * 
            (getAxis(&bhe, otherAxis0) * getAxis(&bhe, otherAxis1) + 
            (split - getAxis(&node->min, axis)) * 
            (getAxis(&bhe, otherAxis0) + getAxis(&bhe, otherAxis1)));
    
    float aboveSA = 
            2 * 
            (getAxis(&bhe, otherAxis0) * getAxis(&bhe, otherAxis1) + 
            (getAxis(&node->max, axis) - split) * 
            (getAxis(&bhe, otherAxis0) + getAxis(&bhe, otherAxis1)));    
        
    float pbelow = belowSA * invTotalSA;
    float pabove = aboveSA * invTotalSA;
    float bonus = 0;//(primAbove == 0 || primBelow == 0) ? 1 : 0;
    float cost = traversalCost + isectCost * (1.0f - bonus) * (pbelow * primBelow + pabove * primAbove);
    return cost;
}

void __device__ splitAABB(AABB* aabb, float split, uint axis, AABB* l, AABB* r)
{
    l->max = aabb->max;
    l->min = aabb->min;
    switch(axis)
    {
    case 0:
        {
            l->max.x = split; r->min.x = split; 
        } break;
    case 1:
        {
            l->max.y = split; r->min.y = split; 
        } break;
    case 2:
        {
            l->max.z = split; r->min.z = split; 
        } break;
    }
}

extern "C" __global__ void computeSplits(AABB* aabb, uint* nodesContent, uint* nodesContentCount, float2* splits, float3* data, uint count, uint depth)
{
    uint aabbStartAddress = (1 << depth) - 1;
    uint id = GlobalId;
    uint sa = 0;
    uint last = 1;
    uint address = count * depth;

    for(uint i = 0; i < (1 << depth); ++i)
    {
        last = nodesContentCount[aabbStartAddress];
        sa += last;
        if(id < sa)
        {
            break;
        }
        aabbStartAddress++;
    }

    AABB aa = aabb[aabbStartAddress];

    uint pos = (id % last);

    address += pos;

    uint add = nodesContent[address];

    if(add == -1) //TODO
    {
        return;
    }

    float3 d = data[add];
    
    int axis = getLongestAxis(aa.max, aa.min);

    float s = getAxis(&d, axis);

    float2 split;
    split.x = getSAH(&aa, axis, s, pos, last - pos - 1);
    split.y = s;

    splits[id] = split;
}

template<typename T>
__device__ void bitonicMergeSortShrd(T* g_values, uint stage)
{
    uint i = GlobalId;

    extern __shared__ T values[];
    
    values[2 * i + 0] = g_values[2 * i + 0];
    values[2 * i + 1] = g_values[2 * i + 1];

    for(uint step = stage >> 1; step > 0; step = step >> 1)
    {
        uint first = (step << 1) * (i/step);
        uint second = first + step;

        if((i % step) > 0)
        {
            first += (i % step);
            second += (i % step);
        }
        
        T n0 = values[first];
           
        T n1 = values[second];
        
        char dir = ((2*i)/stage) % 2;
        char cmp = n0 > n1;
        
        if(dir == 0 && cmp)
        {
            values[first] =  n1;
            values[second] =  n0;
        }
        else if(dir == 1 && !cmp)
        {
            values[first] =  n1;
            values[second] =  n0;
        }
        
        __syncthreads();
    }
    
    g_values[2 * i + 0] = values[2 * i + 0];
    g_values[2 * i + 1] = values[2 * i + 1];
}

extern "C" __global__ void bitonicMergeSortFloatShrd(float* values, uint stage)
{
    bitonicMergeSortShrd<float>(values, stage);
}
