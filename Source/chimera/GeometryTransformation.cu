#pragma once
#include <device_launch_parameters.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include "Transformation.cuh"

#undef USE_SHRD_MEM

__device__ float fx(float x, float y, float z, float amp)
{
    float a = sin(x);
    float b = cos(y);
    float c = sin(z);
    return amp * (a * a + b * b + c * c);
}

extern "C" __global__ void test(VertexData* data, float3* staticNormals, float3* staticPositions, float time, uint N)
{
        uint id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id >= N)
    {
        return;
    }

    VertexData vertexData = data[id];
    float3 pos = staticPositions[id];
    float3 normal = staticNormals[id];
    float3 uvr = pos;

    float freq = 1;
    float2 tc = 2 * vertexData.texCoords + 1; //float2 tc = 2*vertexData.texCoords - 1; WTF?
    float sq = sqrt(tc.x * tc.x + tc.y*tc.y);
    float d = 1;//sin(time + 32 * sq) / sq;
    
    float amp = 0.25;

    float s = fx(4 * freq * uvr.x + time, 2 * freq * uvr.y + time, 8 * uvr.y,  amp);

    float3 dpos = make_float3(normal.x * s, normal.y * s, normal.z * s);

    pos.x = pos.x + dpos.x;// * (d * sin(freq*uvr.x + time) * cos(freq * uvr.y + time));
    pos.y = pos.y + dpos.y;// * (d * sin(freq*uvr.y + time) * cos(freq * uvr.x + time));
    pos.z = pos.z + dpos.z;// * (d * sin(freq*uvr.z + time) * cos(freq * uvr.z + time));
    
    vertexData.position = pos;

    /*
    float2 txt = make_float2(tc.x + 1.0f / 1024.0f, tc.y);
    float2 tyt = make_float2(tc.x, tc.y - 1.0f / 512.0f);

    float3 scx = make_float3(sin(txt.x) * cos(txt.y), sin(txt.x) * sin(txt.y), cos(txt.y));
    float3 scy = make_float3(sin(tyt.x) * cos(tyt.y), sin(tyt.x) * sin(tyt.y), cos(tyt.y));

    float dsx = fx(freq * txt.x + time, freq * tc.y + time, amp);
    float dsy = fx(freq * tc.x + time, freq * tyt.y + time, amp);

    float3 dxpos = scx + make_float3(normal.x * dsx, normal.y * dsx, normal.z * dsx);
    float3 dypos = scy + make_float3(normal.x * dsy, normal.y * dsy, normal.z * dsy);

    vertexData.normal = normalize(-cross(dxpos - dpos, dypos - dpos)); */
    //vertexData.texCoords.x = sin(time * (1+vertexData.texCoords.x));
    data[id] = vertexData;
}

extern "C" __global__ void _sinCosRiddle(VertexData* data, float3* staticNormals, float3* staticPositions, float time, uint N)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id >= N)
    {
        return;
    }

    VertexData vertexData = data[id];
    float3 pos = staticPositions[id];
    float3 normal = staticNormals[id];
    float3 uvr = pos;

    float freq = 1;
    float2 tc = 2 * vertexData.texCoords + 1; //float2 tc = 2*vertexData.texCoords - 1; WTF?
    float sq = sqrt(tc.x * tc.x + tc.y*tc.y);
    float d = 1;//sin(time + 32 * sq) / sq;
    
    float amp = 0.25;

    float s = fx(4 * freq * uvr.x + time, 2 * freq * uvr.y + time, 8 * uvr.y,  amp);

    float3 dpos = make_float3(normal.x * s, normal.y * s, normal.z * s);

    pos.x = pos.x + dpos.x;// * (d * sin(freq*uvr.x + time) * cos(freq * uvr.y + time));
    pos.y = pos.y + dpos.y;// * (d * sin(freq*uvr.y + time) * cos(freq * uvr.x + time));
    pos.z = pos.z + dpos.z;// * (d * sin(freq*uvr.z + time) * cos(freq * uvr.z + time));
    
    vertexData.position = pos;

    /*
    float2 txt = make_float2(tc.x + 1.0f / 1024.0f, tc.y);
    float2 tyt = make_float2(tc.x, tc.y - 1.0f / 512.0f);

    float3 scx = make_float3(sin(txt.x) * cos(txt.y), sin(txt.x) * sin(txt.y), cos(txt.y));
    float3 scy = make_float3(sin(tyt.x) * cos(tyt.y), sin(tyt.x) * sin(tyt.y), cos(tyt.y));

    float dsx = fx(freq * txt.x + time, freq * tc.y + time, amp);
    float dsy = fx(freq * tc.x + time, freq * tyt.y + time, amp);

    float3 dxpos = scx + make_float3(normal.x * dsx, normal.y * dsx, normal.z * dsx);
    float3 dypos = scy + make_float3(normal.x * dsy, normal.y * dsy, normal.z * dsy);

    vertexData.normal = normalize(-cross(dxpos - dpos, dypos - dpos)); */
    //vertexData.texCoords.x = sin(time * (1+vertexData.texCoords.x));
    data[id] = vertexData;
}

extern "C" void sinCosRiddle(VertexData* data, float3* staticNormals, float3* staticPositions, uint gws, uint lws, float time, uint N, cudaStream_t stream)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    //_sinCosRiddle<<<grid, block, 0, stream>>>(data, staticNormals, staticPositions, time, N);
}

extern "C" __global__ void _sinCosRiddle2(VertexData* data, float3* staticNormals, float3* staticPositions, float time, uint N)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id >= N)
    {
        return;
    }

    VertexData vertexData = data[id];
    float3 pos = staticPositions[id];
    float3 normal = staticNormals[id];
    float3 uvr = make_float3(0.5, 0.5, 0.5) + 0.5 * pos;
    //normal = normalize(normal);
    float freq = 8;
    float2 tc = 2*vertexData.texCoords - 1;
    float sq = sqrt(tc.x * tc.x + tc.y*tc.y);
    float d = sin(time + 32 * sq) / sq;
    d = d > 2 ? 2 : d;
    float amp = 0.35;
    float ss = sin(freq * tc.x + time);
    float sc = cos(freq * tc.y + time);

    //ss *= ss;
    //sc *= sc;

    float s = ss * sc;//sc * ss;// * d;//ss * sc;
    s = s;
    pos.x = pos.x;//(d * sin(freq*uvr.x + time) * cos(freq * uvr.y + time));
    pos.y = pos.y + normal.y * amp * s;//(d * sin(freq*uvr.y + time) * cos(freq * uvr.x + time));
    pos.z = pos.z;//(d * sin(freq*uvr.z + time) * cos(freq * uvr.z + time));
    //vertexData.normal = staticNormals[id];
    vertexData.position = pos;
    //vertexData.texCoords.x = sin(time * (1+vertexData.texCoords.x));
    data[id] = vertexData;
}

extern "C" void sinCosRiddle2(VertexData* data, float3* staticNormals, float3* staticPositions, uint gws, uint lws, float time, uint N, cudaStream_t stream)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    //_sinCosRiddle2<<<grid, block, 0, stream>>>(data, staticNormals, staticPositions, time, N);
}

__device__ int sign(int x)
{
    return x < 0 ? -1 : (x > 0 ? 1 : 0);
}

__device__ int GetVertexId(int id, int vertexStride, int gridWidth, int gridSize, int blocksPerRow, int rest, int* _gwm)
{
    int blockId = id / gridSize;

    int blockRow = blockId / blocksPerRow;

    int blockRowId = blockId % blocksPerRow;

    int b = (blockRowId * gridWidth) / (vertexStride - gridWidth);

    int gwm = sign(b) * rest + (1 - sign(b)) * gridWidth;

    *_gwm = gwm;

    if(threadIdx.x >= gwm * gridWidth)
    {
        return -1;
    }

    int s = threadIdx.x / gwm;

    return blockRow * (gridSize * (blocksPerRow-1) + rest * gridWidth) + blockRowId * gridWidth + s * (vertexStride - gwm) + threadIdx.x;
}

__device__ float3 GetPosition(int lid, int id, int vertexStride, int gridWidth, int gridSize, VertexData me, float3* s_D, VertexData* d, uint N)
{
    if(threadIdx.x % gridWidth == 0 && (lid+1) % gridWidth == 0) //border left //same for right
    {
        id -= 1;
        if(id < 0)
        {
            id += N;
        }
        return d[id % N].position;
    }

    if((threadIdx.x+1) % gridWidth == 0 && lid % gridWidth == 0) //border right
    {
        id += 1;
        return d[id % N].position;
    }

    if(lid < 0) //border down
    {
        id -= vertexStride;
        if(id < 0)
        {
            id += N;
        }
        return d[id % N].position;
    }

    if(lid >= gridSize) //border top
    {
        id += vertexStride;
        return d[id % N].position;
    }

    return s_D[lid];
}

extern "C" __global__ void _computeNormals(VertexData* data, int* indices, int vertexStride, int gridWidth, int gridSize, int blocksPerRow, int rest, uint N)
{

#ifdef USE_SHRD_MEM
    extern __shared__ float3 s_D[];
#endif
    
//    int id = gridSize * blockIdx.x + threadIdx.x;
    int id = blockDim.x * blockIdx.x + threadIdx.x;
#ifdef USE_SHRD_MEM
    int ngw;
    int vertexId = GetVertexId(id, vertexStride, gridWidth, gridSize, blocksPerRow, rest, &ngw); //todo, border wrapping
    
    gridSize = ngw * gridWidth;
    gridWidth = ngw;
#else
    int vertexId = id;
#endif

    if(vertexId >= N || vertexId < 0)
    {
        return;
    }

    VertexData vd = data[vertexId];

#ifdef USE_SHRD_MEM
    s_D[threadIdx.x] = vd.position;
    
    __syncthreads();

    /*int topId = threadIdx.x + blockWidth;
    int leftId = threadIdx.x - 1;
    int downId = threadIdx.x - blockWidth;
    int rightId = threadIdx.x + 1; */
#endif

#ifndef USE_SHRD_MEM
    float3 top = data[(vertexId + vertexStride) % N].position;
    float3 down = data[abs((vertexId - vertexStride)) % N].position;
    float3 right = data[(vertexId + 1) % N].position;
    float3 left = data[abs((vertexId - 1)) % N].position;
#else
    float3 top = GetPosition(threadIdx.x + gridWidth, vertexId, vertexStride, gridWidth, gridSize, vd, s_D, data, N);
    float3 down = GetPosition(threadIdx.x - gridWidth, vertexId, vertexStride, gridWidth, gridSize, vd, s_D, data, N);
    float3 right = GetPosition(threadIdx.x + 1, vertexId, vertexStride, gridWidth, gridSize, vd, s_D, data, N);
    float3 left = GetPosition(threadIdx.x - 1, vertexId, vertexStride, gridWidth, gridSize, vd, s_D, data, N);
#endif

    /*float3 me = vd.position;
    float3 metop = normalize(top - me);
    float3 meright = normalize(right - me);
    float3 meleft = normalize(left - me);
    float3 medown = normalize(down - me);

    float3 n0 = cross(metop, meright);
    float3 n1 = cross(meright, medown);
    float3 n2 = cross(medown, meleft);
    float3 n3 = cross(meleft, metop); */

    /*float3 metop = normalize(top - vd.position);
    float3 meright = normalize(right - vd.position);
    float3 meleft = normalize(left - vd.position);
    float3 medown = normalize(down - vd.position); */

    float3 n0 = cross(normalize(top - vd.position), normalize(right - vd.position));
    float3 n1 = cross(normalize(right - vd.position), normalize(down - vd.position));
    float3 n2 = cross(normalize(down - vd.position), normalize(left - vd.position));
    float3 n3 = cross(normalize(left - vd.position), normalize(top - vd.position));

    vd.normal =normalize(n0 + n1 + n2 + n3);
    data[vertexId] = vd;
}

__device__ float3 GetVertex(int localId, float3* s_D)
{
    if(localId < 0 || localId >= 3 * blockDim.x)
    {
        return make_float3(0,0,0);
    }

    return s_D[localId];
}

extern "C" __global__ void _computeNormalsTry2(VertexData* data, int* indices, int vertexStride, int gridWidth, int gridSize, int blocksPerRow, int rest, uint N)
{
    extern __shared__ float3 s_D[];
    
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if(id >= N)
    {
        return;
    }

    int bd = blockDim.x;

    
    VertexData vd = data[id];

    s_D[         threadIdx.x] = data[(id + vertexStride) % N].position; //top
    s_D[    bd + threadIdx.x] = vd.position; //me
    s_D[2 * bd + threadIdx.x] = data[abs(id - vertexStride) % N].position; //down

    __syncthreads();
     
    float3 top = GetVertex(threadIdx.x, s_D);
    float3 down = GetVertex(threadIdx.x + 2 * bd, s_D);
    float3 right = GetVertex(bd + threadIdx.x + 1, s_D);
    float3 left = GetVertex(bd + threadIdx.x - 1, s_D);

    float3 n0 = cross(normalize(top - vd.position), normalize(right - vd.position));
    float3 n1 = cross(normalize(right - vd.position), normalize(down - vd.position));
    float3 n2 = cross(normalize(down - vd.position), normalize(left - vd.position));
    float3 n3 = cross(normalize(left - vd.position), normalize(top - vd.position));

    vd.normal = normalize(n0 + n1 + n2 + n3);
    data[id] = vd;
}

extern "C" void comupteNormals(VertexData* data, int* indices, uint gridDim, uint lws, int gridWidth, int gridSize, int vertexStride, int blocksPerRow, int rest, uint N, uint shrdMemSize, cudaStream_t stream)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gridDim, 1, 1);
    //_computeNormals<<<grid, block, shrdMemSize, stream>>>(data, indices, vertexStride, gridWidth, gridSize, blocksPerRow, rest, N);
    //_computeNormals<<<grid, block, 0, stream>>>(data, indices, vertexStride, gridWidth, gridSize, blocksPerRow, rest, N);
}

extern "C" __global__ void _computeIndices(int* indices, int gridWidth, int gridSize, int vertexStride, int blocksPerRow, int rest)
{
    int id = gridSize * blockIdx.x + threadIdx.x;

    int blockId = id / gridSize;

    int blockRow = blockId / blocksPerRow;

    int blockRowId = blockId % blocksPerRow;

    int b = (blockRowId * gridWidth) / (vertexStride - gridWidth);

    int gwm = sign(b) * rest + (1 - sign(b)) * gridWidth;

    int s = threadIdx.x / gwm;

    indices[id] = blockRow * (gridSize * (blocksPerRow-1) + rest * gridWidth) + blockRowId * gridWidth + s * (vertexStride - gwm) + threadIdx.x;
}

extern "C" void computeIndices(int* indices, int gridDim, int lws, int gridWidth, int gridSize, int vertexStride, int blocksPerRow, int rest, uint N)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gridDim, 1, 1);
    //_computeIndices<<<grid, block, shrdMemSize, stream>>>(data, vertexStride, gridWidth, gridSize, blocksPerRow, rest, N);
}


//b-Splines


__device__ float4 computeVector(float p0, float p1, float p2, float p3)
{
    float4 a;
    a.x = (-p0 + 3 * p1 - 3 * p2 + p3) / 6.0f;
    a.y = (3 * p0 - 6 * p1 + 3 * p2) / 6.0f;
    a.z = (-3 * p0 + 3 * p2) / 6.0f;
    a.w = (p0 + 4 * p1 + p2) / 6.0f;
    return a;
}


__device__ float3 getInterpolPoint(float3* points, const int controlPointsCnt, const int sections)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    int start = id / sections;

    int stride = 1;
    start = ((start - stride) < 0 ) ? (controlPointsCnt - stride) : (start - stride);

    float3 p1 = points[start % controlPointsCnt];
    float3 p2 = points[(start + 1) % controlPointsCnt];
    float3 p3 = points[(start + 2) % controlPointsCnt];
    float3 p4 = points[(start + 3) % controlPointsCnt];
    
    float4 x = computeVector(p1.x, p2.x, p3.x, p4.x);
    float4 y = computeVector(p1.y, p2.y, p3.y, p4.y);
    float4 z = computeVector(p1.z, p2.z, p3.z, p4.z);
    
    float t = (id % sections) / (float)sections;
    float3 pos;
    
    pos.x = t * t * t * x.x + t * t * x.y + t * x.z + x.w;
    pos.y = t * t * t * y.x + t * t * y.y + t * y.z + y.w;
    pos.z = t * t * t * z.x + t * t * z.y + t * z.z + z.w;

    return pos;
}

__device__ float3 getInterpolPointT(float3* points, const int controlPointsCnt, const int sections, int stride)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    
    int start = id / sections;

    start = ((start - stride) < 0 ) ? (controlPointsCnt - (start - stride)) : (start - stride);

    float3 p1 = points[start % controlPointsCnt];
    float3 p2 = points[(start + 1 * stride) % controlPointsCnt];
    float3 p3 = points[(start + 2 * stride) % controlPointsCnt];
    float3 p4 = points[(start + 3 * stride) % controlPointsCnt];
    
    float4 x = computeVector(p1.x, p2.x, p3.x, p4.x);
    float4 y = computeVector(p1.y, p2.y, p3.y, p4.y);
    float4 z = computeVector(p1.z, p2.z, p3.z, p4.z);
    
    float t = (id % sections) / (float)sections;
    float3 pos;
    
    pos.x = t * t * t * x.x + t * t * x.y + t * x.z + x.w;
    pos.y = t * t * t * y.x + t * t * y.y + t * y.z + y.w;
    pos.z = t * t * t * z.x + t * t * z.y + t * z.z + z.w;

    return pos;
}

__device__ float B_SPLINE_KV[16]   = {-1,  3, -3, 1,  
                                       3, -6,  3, 0,  
                                      -3,  0,  3, 0,  
                                       1,  4,  1, 0};

__device__ float B_SPLINE_KV_T[16] = {-1,  3, -3, 1,  
                                       3, -6,  0, 4,  
                                      -3,  3,  3, 1,  
                                       1,  0,  0, 0}; 

/*__device__ float B_SPLINE_KV[16]   = {-1,  3, -3, 1,  
                                       3, -6,  3, 0,  
                                      -3,  3,  0, 0,  
                                       1,  0,  0, 0};

__device__ float B_SPLINE_KV_T[16] = {-1,  3, -3, 1,  
                                       3, -6,  3, 0,  
                                      -3,  3,  0, 0,  
                                       1,  0,  0, 0}; */

__device__ float4 transform(float mat[16], float4 vec)
{
    float4 res;
    res.x = mat [0] * vec.x + mat[1]  * vec.y + mat [2]  * vec.z + mat[3]  * vec.w;
    res.y = mat [4] * vec.x + mat[5]  * vec.y + mat [6]  * vec.z + mat[7]  * vec.w;
    res.z = mat [8] * vec.x + mat[9]  * vec.y + mat [10] * vec.z + mat[11] * vec.w;
    res.w = mat[12] * vec.x + mat[13] * vec.y + mat[14]  * vec.z + mat[15] * vec.w;
    return res;
}

__device__ void mul(float a[16], float b[16], float r[16])
{
    for (int i=0; i<16; i+=4)
    {
        for (int j=0; j<4; j++)
        {
            r[i+j] = b[i]*a[j] + b[i+1]*a[j+4] + b[i+2]*a[j+8] + b[i+3]*a[j+12];
        }
    }
}

__device__ float3 getPoint(float3* points, int dx, int dy, int w, int N, int sections)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    
    int x = (id / sections) % sections;
    int y = ((id / w) / sections) % sections;

    int r = w * (y + dy);
    r = r < 0 ? 0 : r;

    int c = x + dx;
    c = clamp(c, 0, w-1);  
    
    int index = r + c;

    return points[clamp(index, 0, N-1)];
}

__device__ int getOffset(int xx, int yy, int w)
{
    return w * yy + xx;
    //return w * yy + xx;
}
    
extern "C" __global__ void _bspline(float* vertex, float3* points, const uint N, const uint w, const uint controlPointsCnt, const int vstride)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(id > N) return;

    int sections = N / controlPointsCnt;

    //int vstride = 32;

    int sectionsx = w - 3; //w / 4;
    int sectionsy = sectionsx;
            
    float f = vstride / (float)sectionsx;
    int iff = ((int)(f + 0.5f));
    int x = (id % vstride) / iff;
    int y = (id / vstride) / iff;
            
    float v = ((id % (iff))) / (float)(iff - 1);
    float u = 1-((id / (vstride)) % (iff)) / (float)(iff - 1);
    /*int x = (id % vstride) / (vstride / sectionsx);
    int y = (id / vstride) / (vstride / sectionsy); */

    float4 U = make_float4(u * u * u, u * u, u, 1);
    float4 V = make_float4(v * v * v, v * v, v, 1);
    
    int os = getOffset(x, y, w);

    float3 p41 = points[os + 0];//getPoint(points, -2, -2, w, controlPointsCnt, sections);

    float3 p42 = points[os + 1];//getPoint(points, -1, -2, w, controlPointsCnt, sections);

    float3 p43 = points[os + 2];// getPoint(points, 1, -2, w, controlPointsCnt, sections);

    float3 p44 = points[os + 3];//getPoint(points, 2, -2, w, controlPointsCnt, sections);


    float3 p31 = points[os + w + 0];//getPoint(points, -2, -1, w, controlPointsCnt, sections);

    float3 p32 = points[os + w + 1];//getPoint(points, -1, -1, w, controlPointsCnt, sections);

    float3 p33 = points[os + w + 2];//getPoint(points, 1, -1, w, controlPointsCnt, sections);

    float3 p34 = points[os + w + 3];// getPoint(points, 2, -1, w, controlPointsCnt, sections);


    float3 p21 = points[os + 2 * w + 0];//getPoint(points, -2, 1, w, controlPointsCnt, sections);

    float3 p22 = points[os + 2 * w + 1];// getPoint(points, -1, 1, w, controlPointsCnt, sections);

    float3 p23 = points[os + 2 * w + 2];// getPoint(points, 1, 1, w, controlPointsCnt, sections);

    float3 p24 = points[os + 2 * w + 3];//getPoint(points, 2, 1, w, controlPointsCnt, sections);


    float3 p11 = points[os + 3 * w + 0];//getPoint(points, -2, 2, w, controlPointsCnt, sections);

    float3 p12 = points[os + 3 * w + 1];//getPoint(points, -1, 2, w, controlPointsCnt, sections);

    float3 p13 = points[os + 3 * w + 2];// getPoint(points, 1, 2, w, controlPointsCnt, sections);

    float3 p14 = points[os + 3 * w + 3];// getPoint(points, 2, 2, w, controlPointsCnt, sections);


    float psx[16] = {p11.x, p12.x, p13.x, p14.x,  p21.x, p22.x, p23.x, p24.x,  p31.x, p32.x, p33.x, p34.x, p41.x, p42.x, p43.x, p44.x};
    float psy[16] = {p11.y, p12.y, p13.y, p14.y,  p21.y, p22.y, p23.y, p24.y,  p31.y, p32.y, p33.y, p34.y, p41.y, p42.y, p43.y, p44.y};
    float psz[16] = {p11.z, p12.z, p13.z, p14.z,  p21.z, p22.z, p23.z, p24.z,  p31.z, p32.z, p33.z, p34.z, p41.z, p42.z, p43.z, p44.z};

    U *= 1.0 / 6.0;
    V *= 1.0 / 6.0;

    float4 pos = make_float4(0,0,0,0);

    float4 tx = transform(B_SPLINE_KV_T, V);
    tx = transform(psx, tx);
    tx = transform(B_SPLINE_KV, tx);
    pos.x = dot(U, tx);

    float4 ty = transform(B_SPLINE_KV_T, V);
    ty = transform(psy, ty);
    ty = transform(B_SPLINE_KV, ty);
    pos.y = dot(U, ty);

    float4 tz = transform(B_SPLINE_KV_T, V);
    tz = transform(psz, tz);
    tz = transform(B_SPLINE_KV, tz);
    pos.z = dot(U, tz);

    const int vs = 8;
     
    vertex[vs * id + 0] = pos.x;
    vertex[vs * id + 1] = pos.y;
    vertex[vs * id + 2] = pos.z; 
}

extern "C" void bspline(int gws, int lws, VertexData* data, float3* points, uint N, uint controlPointsCnt, uint w, uint vstride, cudaStream_t stream)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    //_bspline<<<grid, block, 0, stream>>>((float*)data, points, N, w, controlPointsCnt, vstride);
}

extern "C" __global__ void _animateBSline(float3* points, const float time, uint N, uint w, uint h)
{
    uint id = blockDim.x * blockIdx.x + threadIdx.x;
    uint xi = id % h;
    uint yi = id / w;

    if(id > N) return;
    
    float3 p = points[id];
      
    float x = -1 + 2 * xi / (float)w;
    float y = -1 + 2 * yi / (float)h;

    float sq = -sqrt(x*x + y*y + 0.1);
    float freq = 20;
    p.y = 0.5 + 0.5 * sin(time + freq * sq) / sq;
    points[id] = p;
}

extern "C" void animateBSline(int gws, int lws, float3* points, float time, uint N, uint w, uint h, cudaStream_t stream)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    //_animateBSline<<<grid, block, 0, stream>>>(points, time, N, w, h);
}

__device__ float GX[9] = {1,0,-1, 2,0,-2, 1,0,-1};
__device__ float GY[9] = {1,2,-1, 0,0,0, -1,-2,-1};

extern "C" __global__ void _computeNormalsBSplines(VertexData* data, int vertexStride, uint N)
{

    int vertexId = blockDim.x * blockIdx.x + threadIdx.x;

    if(vertexId >= N || vertexId < 0)
    {
        return;
    }

    VertexData vd = data[vertexId];

    float3 top = data[(vertexId + (2*vertexStride)) % N].position;
    float3 down = data[abs((vertexId - (2*vertexStride))) % N].position;
    float3 right = data[(vertexId + 2) % N].position;
    float3 left = data[abs((vertexId - 2)) % N].position;
     
    float3 n0 = cross(normalize(top - vd.position), normalize(right - vd.position));
    float3 n1 = cross(normalize(right - vd.position), normalize(down - vd.position));
    float3 n2 = cross(normalize(down - vd.position), normalize(left - vd.position));
    float3 n3 = cross(normalize(left - vd.position), normalize(top - vd.position));

    /*
    float leftUp = data[abs((vertexId + vertexStride - 1)) % N].position.y;
    float leftDown = data[abs((vertexId - vertexStride - 1)) % N].position.y;
    float rightUp = data[abs((vertexId + vertexStride + 1)) % N].position.y;
    float rightDown = data[abs((vertexId - vertexStride + 1)) % N].position.y;


    float gx = leftUp - rightUp + 2 * left - 2 * right + leftDown - rightDown;
    float gy = leftUp + 2 * top + rightUp - leftDown - 2 * down - rightDown;
    float z  = sqrt(gx * gx + gy * gy);
    float3 n = normalize(make_float3(2 * gx, 1, 2 * gy));
    n.z = 1 - n.z;
    vd.normal = normalize(n);*/

    vd.normal = normalize(n0 + n1 + n2 + n3);
    data[vertexId] = vd;
}

extern "C" void comupteNormalsBSpline(VertexData* data, uint gws, uint lws, uint vertexStride, uint N, cudaStream_t stream)
{
    dim3 block(lws, 1, 1);
    dim3 grid(gws / block.x, 1, 1);
    //_computeNormalsBSplines<<<grid, block, 0, stream>>>(data, vertexStride, N);
}

