#pragma once
#include <device_launch_parameters.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include "Particles.cuh"

#define CHECK_ALIVE(pos) \
    if(!IsAlive(&pos))  \
    { \
        return; \
    }

#define CHECK_IN (!IsOut(&pos, &min, &max)) 
   
typedef unsigned int UINT;  
 
#define GlobalId blockDim.x * blockIdx.x + threadIdx.x
 
texture<float4, cudaTextureType3D, cudaReadModeElementType> ct_gradientTexture;
 
__device__ int IsOut(float4* pos, float3* min, float3* max)
{ 
    return  
        min->x > pos->x || max->x < pos->x ||
        min->y > pos->y || max->y < pos->y ||
        min->z > pos->z || max->z < pos->z;
} 

__device__ bool IsAlive(float4* pos)
{
    return (pos->w > 0.5) && (pos->w < 1.5);
}

__device__ bool WasDead(float4* pos)
{
    return (pos->w > 1.5) && (pos->w < 2.5);
}

extern "C" __global__ void _computeEmitter(float4* positions, float4* startingPositions, float3* velos, float3* acc, EmitterData* ed, float3 translation, float time, float dt, float start, float end, UINT N)
{
    UINT id = GlobalId;
    float4 pos = positions[id];
    EmitterData data = ed[id];

    void* d = malloc(10);

    data.time += dt;

    if(IsAlive(&pos) && (time - data.birthTime) > end)
    {
        pos = startingPositions[id];
        pos.w = 0;
        positions[id] = pos;
        //data.time = 0.0;
        data.birthTime = time;
        ed[id] = data;
        acc[id] = make_float3(0,0,0);
        velos[id] = make_float3(0,0,0);
    }
    
    if(!IsAlive(&pos) && data.time > end * data.rand)
    {
        pos.x += translation.x;
        pos.y += translation.y;
        pos.z += translation.z;
        pos.w = 1.0;
        data.birthTime = time;
        ed[id] = data;
        positions[id] = pos;
        return;
    }

    ed[id] = data;



    /*
    if(!IsAlive(&pos))
    {
        data.time += dt;
        ed[id] = data;
    }
  
    if(IsAlive(&pos) && (time - data.birthTime) > end)
    {
        pos = startingPositions[id];
        pos.w = 2;
        /*
        positions[id] = pos;
        data.time = 0.0;
        data.birthTime = time;
        ed[id] = data;
        acc[id] = make_float3(0,0,0);
        velos[id] = make_float3(0,0,0); */

    /*} else if(pos.w < 0.5)
    {
        if(data.time > end * data.rand)
        {
            pos.x += translation.x;
            pos.y += translation.y;
            pos.z += translation.z;
            pos.w = 1.0;
            data.birthTime = time;
            ed[id] = data;
            positions[id] = pos;
        }
    }
    
    if(WasDead(&pos))
    {
        pos.x += translation.x;
        pos.y += translation.y;
        pos.z += translation.z;
        pos.w = 3.0;
        data.birthTime = time;
        data.time = 0.0;
        ed[id] = data;
        positions[id] = pos;
    } */
}

extern "C" void computeEmitter(float4* positions, float4* startingPositions, float3* velos, float3* acc, EmitterData* ed, float3 translation, float time, float dt, float start, float end, UINT gws, UINT lws)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    _computeEmitter<<<grid, block>>>(positions, startingPositions, velos, acc, ed, translation, time, dt, start, end, gws);
}


extern "C" __global__ void _computeGravity(float4* positions, float3* acc, float factor)
{
    UINT id = GlobalId;
    float4 pos = positions[id];
    
    CHECK_ALIVE(pos);

    //if(CHECK_IN)
    {
        float3 a = acc[id];
        a.y += factor;
        acc[id] = a;
    } 
}

extern "C" void computeGravity(float4* positions, float3* acc, float factor, UINT gws, UINT lws)
{
    dim3 block(lws, 1, 1); 
    dim3 grid(gws / block.x, 1, 1);
    _computeGravity<<<grid, block>>>(positions, acc, factor); 
}

extern "C" __global__ void _computeTurbulence(float4* positions, float3* acc, float3* dirs, UINT randomCount, UINT time)
{
    UINT id = GlobalId;
    float4 pos = positions[id];
    
    CHECK_ALIVE(pos);

    //if(CHECK_IN)
    {
        float3 a = acc[id];
        UINT dirsIndex = (id + time) % randomCount;
        float3 dir = dirs[dirsIndex];
        a.x += dir.x;
        a.y += dir.y;
        a.z += dir.z;
        acc[id] = a;
    }
}

extern "C" void computeTurbulence(float4* positions, float3* acc, float3* dirs, UINT randomCount, UINT time, UINT gws, UINT lws)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    _computeTurbulence<<<grid, block>>>(positions, acc, dirs, randomCount, time);
}

extern "C" __global__ void _integrate(float4* positions, float3* acc, float3* velocity, float dt)
{ 
    UINT id = GlobalId;

    float4 pos = positions[id];

    float3 a = acc[id];
    float3 v = velocity[id];
    

    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;

    pos.x += v.x * dt;
    pos.y += v.y * dt;
    pos.z += v.z * dt;

    velocity[id] = v;
    positions[id] = pos;
}

extern "C" void integrate(float4* positions, float3* acc, float3* velocity, float dt, UINT gws, UINT lws)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    _integrate<<<grid, block>>>(positions, acc, velocity, dt);
}

extern "C" __global__ void _computeGradientField(float4* positions, float3* velo, float4 position)
{
    UINT id = GlobalId;
    float4 pos = positions[id];

    CHECK_ALIVE(pos);

    pos *= 30;

    float3 coord = make_float3(pos.x, pos.y, pos.z);
    float scale = position.w;
    position = make_float4(1,5,8,0);
    
    float3 dist = make_float3(pos.x - position.x, pos.y - position.y, pos.z - position.z);

    float distanceSquared = dot(dist, dist);

    float range = 30; 

    /*if(distanceSquared > range * range)
    {
        return; 
    } */

    float4 grad = tex3D(ct_gradientTexture, position.x + coord.x * scale, position.y + coord.y * scale, position.z + coord.z * scale);

    float t = 5;
//    float s = t - t * (distanceSquared / (range * range));

    grad *= 6;// * s; 

    float3 v = velo[id];

    v.x = grad.x;
    v.y = grad.y;
    v.z = grad.z;
    velo[id] = v;
}

extern "C" void computeGradientField(float4* positions, float3* acc, float4 pos, UINT gws, UINT lws)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    _computeGradientField<<<grid, block>>>(positions, acc, pos);
}


extern "C" __global__ void _computeGravityField(float4* positions, float3* acc, float4 posNrange, int repel, float scale)
{
    UINT id = GlobalId;
    float4 pos = positions[id];

    CHECK_ALIVE(pos);

    float3 gpos = make_float3(posNrange.x, posNrange.y, posNrange.z);

    float range = 5 * posNrange.w;

    float3 grad = make_float3(pos.x - gpos.x, pos.y - gpos.y, pos.z - gpos.z);

    float distanceSquared = dot(grad, grad);

    if(distanceSquared > range * range)
    {
        return;
    }
    
    float s = repel - (distanceSquared / (range * range));
    s *= scale;
    //grad = normalize(grad);
    grad *= s;
    
    float3 a = acc[id];
    a += grad;
    acc[id] = a;
}


extern "C" void computeGravityField(float4* positions, float3* acc, float4 posNrange, int repel, float scale, UINT gws, UINT lws)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    _computeGravityField<<<grid, block>>>(positions, acc, posNrange, repel, scale);
}

extern "C" __global__ void _computeVelocityDamping(float4* positions, float3* velo, float damping)
{
    UINT id = GlobalId;
    float4 pos = positions[id];

    CHECK_ALIVE(pos);

    float3 a = velo[id];
    a *= damping;
    velo[id] = a;
}

extern "C" void computeVelocityDamping(float4* positions, float3* velo, float damping, UINT gws, UINT lws)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    _computeVelocityDamping<<<grid, block>>>(positions, velo, damping);
}


extern "C" void bindGradientTexture(const cudaArray* array, const cudaChannelFormatDesc desc)
{
    ct_gradientTexture.addressMode[0] = cudaAddressModeMirror;
    ct_gradientTexture.addressMode[1] = cudaAddressModeMirror;
    ct_gradientTexture.addressMode[2] = cudaAddressModeMirror;
    ct_gradientTexture.normalized = 0;
    ct_gradientTexture.filterMode = cudaFilterModeLinear;

    cudaBindTextureToArray(ct_gradientTexture, array, desc);
}

typedef float4 Plane;

__device__ float getDistance(Plane p, float4 position)
{
    return dot(make_float3(position), make_float3(p)) - p.w;
}

__device__  bool isOutside(Plane p, float4 position)
{
    return getDistance(p, position) < 0;
}

extern "C" __global__ void _computePlane(Plane p, float3* velos, float4* positions)
{
    UINT id = GlobalId;
    float4 pos = positions[id];

    CHECK_ALIVE(pos);

    if(isOutside(p, pos))
    {
        float3 v = velos[id];
        v = -v;
        pos.x = pos.x + v.x;
        pos.y = pos.y + v.y;
        pos.z = pos.z + v.z;
        /*velos[id] = v;
        positions[id] = pos; */
    }
}

template<typename T>
__device__ void _reduce(float4* data, float3* dst, T _operator)
{
    extern __shared__ float3 s_d[];

    UINT id = threadIdx.x;
    UINT si = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    float4 d0 = data[si];
    float4 d1 = data[si + blockDim.x];

    s_d[id] = _operator(make_float3(d0.x, d0.y, d0.z), make_float3(d1.x, d1.y, d1.z));
    
    __syncthreads();
    
    for(UINT i = blockDim.x/2 ; i > 0; i >>= 1)
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

typedef float3 (*function)(float3, float3);

extern "C" __global__ void _reduce_max4(float4* data, float3* dst)
{
    _reduce<function>(data, dst, fmaxf);
}

extern "C" __global__ void _reduce_min4(float4* data, float3* dst)
{
    _reduce<function>(data, dst, fminf);
}

extern "C" __global__ void _scanKD(uint* content, uint neutralItem, uint depth)
{
    extern __shared__ uint shrdMem[];

    uint thid = threadIdx.x;
    uint grpId = blockIdx.x;
    uint N = 2 * blockDim.x;
    uint offset = 1;
    
    uint i0 = content[2*thid + grpId * N];
    uint i1 = content[2*thid + 1 + grpId * N];

    if(neutralItem == i0)
    {
        shrdMem[2 * thid  ] = 0;
    }
    else
    {
        shrdMem[2 * thid  ] = 1;
    }

    if(neutralItem == i1)
    {
        shrdMem[2 * thid + 1] = 0;
    }
    else
    {
        shrdMem[2 * thid + 1] = 1;
    }

    //uint last = shrdMem[2*thid+1];

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
        content[shrdMem[2*thid]] = i0;
    }

    if(neutralItem != i1)
    {
        content[shrdMem[2*thid+1]] = i1;
    }

    /*content[shrdMem[2*thid]] = i0;
    content[shrdMem[2*thid+1]] = i1; */
    
    /*if(writeSums && (thid == get_local_size(0)-1))
    {
        sums[grpId] = shrdMem[2*thid+1] + last;
    } */
}

__device__ float getAxis(float4* vec, UINT axis)
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

__device__ float getAxis(float3* vec, uint axis)
{
    float4 v = make_float4(vec->x, vec->y, vec->z, 0);
    return getAxis(&v, axis);
}

extern "C" __global__ void kdTree(uint* nodesContent, void* splits, float4* data, uint count, uint depth)
{
    uint id = GlobalId;
    uint offset = 1 << depth;
    uint axis = *((uint*)splits);
    float split = *(((float*)splits)+4);
    
    float4 d = data[id];

    if(split < getAxis(&d, axis))
    {
        nodesContent[offset + count + id] = id;
    }
    else
    {
        nodesContent[offset + 2 * count + id] = id;
    }
}
