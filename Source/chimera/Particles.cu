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
 
#define GlobalId (blockDim.x * blockIdx.x + threadIdx.x)
 
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

extern "C" __global__ 
    void _computeEmitter(
    float4* positions, 
    float4* startingPositions, 
    float3* velos, 
    float3* acc, 
    EmitterData* ed, 
    float3 translation, 
    float time, 
    float dt, 
    float start, 
    float end, 
    uint N)
{
    uint id = GlobalId;

    if(id >= N) return;

    float4 pos = positions[id];
    EmitterData data = ed[id];

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
}

extern "C" __global__ void _computeGravity(float4* positions, float3* acc, float3 dir, uint N)
{
    uint id = GlobalId;

    if(id >= N) return;

    float4 pos = positions[id];
    
    CHECK_ALIVE(pos);

    //if(CHECK_IN)
    {
        float3 a = acc[id];
        a += dir;
        acc[id] = a;
    } 
}

extern "C" __global__ void _computeTurbulence(float4* positions, float3* acc, float3* dirs, uint randomCount, float time, uint N)
{
    uint id = GlobalId;

    if(id >= N) return;

    float4 pos = positions[id];

    CHECK_ALIVE(pos);

    //if(CHECK_IN)
    {
        float3 a = acc[id];
        uint dirsIndex = (id + (uint)time) % randomCount;
        float3 dir = dirs[dirsIndex];
        a.x += dir.x;
        a.y += dir.y;
        a.z += dir.z;
        acc[id] = a;
    }
}

extern "C" __global__ void _integrate(float4* positions, float3* acc, float3* velocity, float dt, uint N)
{ 
    uint id = GlobalId;

    if(id >= N) return;

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

extern "C" __global__ void _computeGradientField(float4* positions, float3* velo, float4 position, uint N)
{
    uint id = GlobalId;
    if(id >= N) return;

    float4 pos = positions[id];

    CHECK_ALIVE(pos);

    float3 coord = make_float3(pos.x, pos.y, pos.z);
    float scale = 1.0/32.0; //position.w;
    coord *= scale;
    
    float3 dist = make_float3(pos.x - position.x, pos.y - position.y, pos.z - position.z);

    float distanceSquared = dot(dist, dist);

    float range = 30; 

    /*if(distanceSquared > range * range)
    {
        return; 
    } */

    float4 grad = tex3D(ct_gradientTexture, coord.x, coord.y, coord.z);

    float t = 5;

//    float s = t - t * (distanceSquared / (range * range));

    grad *= 4;// * s; 

    float3 v = velo[id];

    v.x = grad.x;
    v.y = grad.y;
    v.z = grad.z;
    velo[id] = v;
}

extern "C" __global__ void _computeGravityField(float4* positions, float3* acc, float4 posNrange, int repel, float scale)
{
    uint id = GlobalId;
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

extern "C" __global__ void _computeVelocityDamping(float4* positions, float3* velo, float damping)
{
    uint id = GlobalId;
    float4 pos = positions[id];

    CHECK_ALIVE(pos);

    float3 a = velo[id];
    a *= damping;
    velo[id] = a;
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
    uint id = GlobalId;
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