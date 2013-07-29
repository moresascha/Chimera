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

__device__ int IsAlive(float4* pos)
{
    return pos->w > 0.5;
}

extern "C" __global__ void _computeEmitter(float4* positions, float4* startingPositions, float3* velos, float3* acc, EmitterData* ed, float3 translation, float time, float dt, float start, float end, UINT N)
{
    UINT id = GlobalId;
    float4 pos = positions[id];
    EmitterData data = ed[id];

    //float4 copy = pos;

    if(!IsAlive(&pos))
    {
        data.time += dt;
        ed[id] = data;
    }
  
    if((time - data.birthTime) > end)//if(IsOut(&copy, &min, &max))
    {
        pos = startingPositions[id];
        positions[id] = pos;
        data.time = 0;
        data.birthTime = time;
        ed[id] = data;
        acc[id] = make_float3(0,0,0);
        velos[id] = make_float3(0,0,0);
        return;

    } else if(!IsAlive(&pos))
    {
        if(data.time > end * data.rand)
        {
            pos.x += translation.x;
            pos.y += translation.y;
            pos.z += translation.z;
            pos.w = 1;
            data.birthTime = time;
            ed[id] = data;
            positions[id] = pos;
        }
    }
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
    

    v.x += a.x * dt * dt;
    v.y += a.y * dt * dt;
    v.z += a.z * dt * dt;

    pos.x += v.x * dt * dt;
    pos.y += v.y * dt * dt;
    pos.z += v.z * dt * dt;

    velocity[id] = v;
    positions[id] = pos;
}

extern "C" void integrate(float4* positions, float3* acc, float3* velocity, float dt, UINT gws, UINT lws)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    _integrate<<<grid, block>>>(positions, acc, velocity, dt);
}

extern "C" __global__ void _computeGradientField(float4* positions, float3* acc, float4 position)
{
    UINT id = GlobalId;
    float4 pos = positions[id];

    CHECK_ALIVE(pos);

    float3 coord = make_float3(pos.x, pos.y, pos.z);
    float scale = position.w;
    float4 grad = tex3D(ct_gradientTexture, position.x + coord.x * scale, position.y + coord.y * scale, position.z + coord.z * scale);

    float3 dist = make_float3(pos.x - position.x, position.y - position.y, pos.z - position.z);

    float distanceSquared = dot(dist, dist);

    float range = 30;

    if(distanceSquared > range * range)
    {
        return; 
    }

    float t = 5;
    float s = t - t * (distanceSquared / (range * range));

    grad *= s;

    float3 a = acc[id];
    a.x += grad.x;
    a.y += grad.y;
    a.z += grad.z;
    acc[id] = a;
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
    ct_gradientTexture.normalized = 1;
    ct_gradientTexture.filterMode = cudaFilterModeLinear;

    cudaBindTextureToArray(ct_gradientTexture, array, desc);
}
