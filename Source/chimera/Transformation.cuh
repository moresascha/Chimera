#pragma once
typedef unsigned int uint;
struct VertexData
{
    float3 position;
    float3 normal;
    float2 texCoords;
};

extern "C" void sinCosRiddle(VertexData* data, float3* staticNormals, float3* staticPositions, uint gws, uint lws, float time, uint N, cudaStream_t stream);

extern "C" void sinCosRiddle2(VertexData* data, float3* staticNormals, float3* staticPositions, uint gws, uint lws, float time, uint N, cudaStream_t stream);

extern "C" void comupteNormals(VertexData* data, int* indices, uint gridDim, uint lws, int gridWidth, int gridSize, int vertexStride, int blocksPerRow, int rest, uint N, uint shrdMemSize, cudaStream_t stream);

extern "C" void computeIndices(int* indices, int gridDim, int lws, int gridWidth, int gridSize, int vertexStride, int blocksPerRow, int rest, uint N);

extern "C" void bspline(int gws, int lws, VertexData* data, float3* points, uint N, uint controlPointsCnt, uint w, uint vstride, cudaStream_t stream);

extern "C" void animateBSline(int gws, int lws, float3* points, float time, uint N, uint w, uint h, cudaStream_t stream);

extern "C" void comupteNormalsBSpline(VertexData* data, uint gws, uint lws, uint vertexStride, uint N, cudaStream_t stream);
