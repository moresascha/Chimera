#pragma once
#include "kdtree.cuh"

#define GlobalId (blockDim.x * blockIdx.x + threadIdx.x)

#define INVALID_DATA_ADD ((uint)-1)
#define FLT_MAX 3.402823466e+38F

typedef float3 (*function)(float3, float3);


template<typename _SM_T, typename _SRC_T, typename _DST_T, typename _OPERATOR>
__device__ void __reduce(_SM_T s_d, _SRC_T d0, _SRC_T d1, _DST_T dst, _OPERATOR _operator)
{
	uint id = threadIdx.x;

    s_d[id] = _operator(make_float3(d0.x, d0.y, d0.z), make_float3(d1.x, d1.y, d1.z));
	
	__syncthreads();
	
	for(uint i = blockDim.x/2 ; i > 0; i >>= 1)
	{
		if(id < i)
		{
			s_d[id] = _operator(s_d[id + i], s_d[id]);
		}

        __syncthreads();
	}

	if(id == 0)
	{
		dst[blockIdx.x] = s_d[0];
	} 
}

template<typename _OPERATOR, typename _SRC_T, typename _DST_T>
__device__ void _reduce(_SRC_T* data, _DST_T* dst, _OPERATOR _operator, uint stride, uint memoryOffset)
{
    extern __shared__ _DST_T s_d[];

	uint si = blockIdx.x * stride + threadIdx.x;

	_SRC_T d0 = data[memoryOffset + si];
    _SRC_T d1 = data[memoryOffset + si + blockDim.x];

    __reduce<_DST_T*, _SRC_T, _DST_T*, _OPERATOR>(s_d, d0, d1, dst, _operator);
}

template<typename _OPERATOR, typename _SRC_T, typename _DST_T>
__device__ void _reduceFromIndex(_SRC_T* data, _DST_T* dst, _OPERATOR _operator, uint* index, _SRC_T extremes, uint stride, uint memoryOffset)
{
    extern __shared__ _DST_T s_d[];
    
	uint si = blockIdx.x * stride + threadIdx.x;

    uint i0 = index[memoryOffset + si];
    uint i1 = index[memoryOffset + si + blockDim.x];

	_SRC_T d0;
    _SRC_T d1;

    if(i0 == INVALID_DATA_ADD)
    {
        d0 = extremes;
    }
    else
    {
        d0 = data[i0];
    }

    if(i1 == INVALID_DATA_ADD)
    {
        d1 = extremes;
    }
    else
    {
        d1 = data[i1];
    }

    __reduce<_DST_T*, _SRC_T, _DST_T*, _OPERATOR>(s_d, d0, d1, dst, _operator);
}

extern "C" __global__ void _reduce_fromIndexMin4to3(float4* data, float3* dst, uint* index, uint stride, uint memoryOffset)
{
    _reduceFromIndex<function, float4, float3>(data, dst, fminf, index, make_float4(FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX), stride, memoryOffset);
}

extern "C" __global__ void _reduce_fromIndexMax4to3(float4* data, float3* dst, uint* index, uint stride, uint memoryOffset)
{
    _reduceFromIndex<function, float4, float3>(data, dst, fmaxf, index, make_float4(-FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX), stride, memoryOffset);
}

extern "C" __global__ void _reduce_max4to3(float4* data, float3* dst, uint stride, uint offset)
{
    _reduce<function, float4, float3>(data, dst, fmaxf, stride, offset);
}

extern "C" __global__ void _reduce_min4to3(float4* data, float3* dst, uint stride, uint offset)
{
    _reduce<function, float4, float3>(data, dst, fminf, stride, offset);
}

extern "C" __global__ void _reduce_max3to3(float3* data, float3* dst, uint stride, uint offset)
{
    _reduce<function, float3, float3>(data, dst, fmaxf, stride, offset);
}

extern "C" __global__ void _reduce_min3to3(float3* data, float3* dst, uint stride, uint offset)
{
    _reduce<function, float3, float3>(data, dst, fminf, stride, offset);
}

//todo: template it
extern "C" __global__ void _scanKD(uint* content, uint* contentCount, uint neutralItem, uint memoryOffset, uint depth)
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
        contentCount[((1 << (depth+1))-1) + grpId] = shrdMem[2*thid+1] + last;
    }
}

__device__ SplitData* getParentSplit(SplitData* splitData, uint* nodesContent, uint* address, uint depth)
{
    return 0;
}

extern "C" __global__ void spreadContent(uint* nodesContent, uint* nodesContentCount, SplitData* splitData, float4* data, uint count, uint offset, uint depth)
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

    if(add == -1) //TODO
    {
        return;
    }

    float4 d = data[add];

    uint memoryOffset = count * ((1 << (depth+1)) - 1) + 2 * m * count;

    if(sd.split > getAxis(&d, sd.axis))
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

extern "C" __global__ void bla()
{

}

extern "C" __global__ void computeSplits(AABB* aabb, uint* nodesContent, uint* nodesContentCount, SplitData* splitData, float4* data, uint count, uint depth)
{
/*
    uint aabbStartAddress = (1 << depth) - 1;
    uint id = threadIdx.x;
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

    address += (id % last);

    uint add = nodesContent[address];

    if(add == -1) //TODO
    {
        return;
    }

    float4 d = data[add];

    uint axis = blockIdx.x;

    float split = getAxis(&d, axis);
    
    AABB l; AABB r;

    splitAABB(&aa, split, axis, &l, &r);*/
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
