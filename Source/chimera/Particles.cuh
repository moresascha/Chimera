#pragma once
typedef unsigned int UINT;

struct EmitterData
{
    float rand;
    float birthTime;
    float time;
    float tmp;
};

/*extern "C" void computeEmitter(float4* positions, float4* startingPositions, float3* velos, float3* acc, EmitterData* ed, float3 translation, float time, float dt, float start, float end, UINT gws, UINT lws);

extern "C" void computeGravity(float4* positions, float3* acceleration, float factor, UINT gws, UINT lws);

extern "C" void computeTurbulence(float4* positions, float3* acc, float3* dirs, UINT randomCount, UINT time, UINT gws, UINT lws);

extern "C" void computeGradientField(float4* positions, float3* acc, float4 position, UINT gws, UINT lws);

extern "C" void computeGravityField(float4* positions, float3* acc, float4 positionNrange, int repel, float scale, UINT gws, UINT lws);

extern "C" void computeVelocityDamping(float4* positions, float3* velocity, float damping, UINT gws, UINT lws);

extern "C" void integrate(float4* positions, float3* acceleration, float3* velocity, float dt, UINT gws, UINT lws); */

extern "C" void bindGradientTexture(const cudaArray* array, const cudaChannelFormatDesc desc);

