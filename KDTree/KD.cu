#pragma once
#include "kdtree.cuh"

#define GlobalId (blockDim.x * blockIdx.x + threadIdx.x)

#define INVALID_ADDRESS ((uint)-1)
#define FLT_MAX 3.402823466e+38F
#define FLT_MAX_DIV2 (FLT_MAX/2.0f)
#define PI 3.14159265359
#define PI_MUL2 (2 * PI)


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

struct DstDesc
{
    int isright;
    uint prefixSum;
    Split s;
};

__device__ DstDesc getDstNode(Split* splitData, uint* scannedContentCount, uint* prefixContent, uint depth)
{
    uint id = GlobalId;
    
    uint startAdd = elemsBeforeLevel(depth);

    uint index = scannedContentCount[id];

    Split sd = splitData[startAdd + index];
    
    uint prefixSum = prefixContent[id];
    
    uint modi = id - prefixSum;
    
    int right = modi >= sd.below;

    DstDesc d;
    d.isright = right;
    d.prefixSum = prefixSum;
    d.s = sd;

    return d;
}

__device__ uint getNextScanValue(uint* scannedContentCount, int isright, uint depth)
{
    uint id = GlobalId;

    uint scanVal = scannedContentCount[id];

    scanVal = 2 * scanVal;//(scanVal > 0) + !(scanVal & 1);

    return scanVal + isright;
}

extern "C" __global__ void spreadContent(uint* nodesContent, uint* nodesContentCount, Split* splitData, uint count, uint depth, 
                                         uint* scannedContentCount,
                                         uint* prefixScan)
{
    uint id = GlobalId;

    if(id >= 2*count)
    {
        return;
    }

    DstDesc dd = getDstNode(splitData, scannedContentCount, prefixScan, depth);

    uint nextScanVal = getNextScanValue(scannedContentCount, dd.isright, depth);

//     uint memoryOffset = count * ((1 << (depth + 1)) - 1) + scanVal * count;
//     uint osright = memoryOffset + id - sd.below - prefixSum;
//     uint osleft = memoryOffset + id;

    if(!dd.isright)
    {
        prefixScan[id] = dd.prefixSum;
    }
    else
    {
        prefixScan[id] = dd.prefixSum + dd.s.below;
    }

    scannedContentCount[id] = nextScanVal;
}

float3 __device__ getAxisScale(AABB* aabb)
{
    return make_float3(aabb->max.x - aabb->min.x, aabb->max.y - aabb->min.y, aabb->max.z - aabb->min.z);
}

float __device__ getArea(AABB* aabb)
{
    float3 axisScale = getAxisScale(aabb);
    return 2 * axisScale.x * axisScale.y + 2 * axisScale.x * axisScale.z + 2 * axisScale.y * axisScale.z;
}

float __device__ getSAH(AABB* node, int axis, float split, int primBelow, int primAbove, float bonus, float traversalCost = 0, float isectCost = 1)
{
    float cost = FLT_MAX;
    if(split > getAxis(&node->min, axis) && split < getAxis(&node->max, axis))
    {
        float3 axisScale = getAxisScale(node);
        float invTotalSA = 1.0f / getArea(node);
        int otherAxis0 = (axis+1) % 3;
        int otherAxis1 = (axis+2) % 3;
        float belowSA = 
            2 * 
            (getAxis(&axisScale, otherAxis0) * getAxis(&axisScale, otherAxis1) + 
            (split - getAxis(&node->min, axis)) * 
            (getAxis(&axisScale, otherAxis0) + getAxis(&axisScale, otherAxis1)));
    
        float aboveSA = 
            2 * 
            (getAxis(&axisScale, otherAxis0) * getAxis(&axisScale, otherAxis1) + 
            (getAxis(&node->max, axis) - split) * 
            (getAxis(&axisScale, otherAxis0) + getAxis(&axisScale, otherAxis1)));    
        
        float pbelow = belowSA * invTotalSA;
        float pabove = aboveSA * invTotalSA;
        //float bonus = 0;//(primAbove == 0 || primBelow == 0) ? 1 : 0;
        cost = traversalCost + isectCost * (1.0f - bonus) * (pbelow * primBelow + pabove * primAbove);
    }
    return cost;
}

void __device__ splitAABB(AABB* aabb, float split, uint axis, AABB* l, AABB* r)
{
    l->max = aabb->max;
    l->min = aabb->min;
    r->max = aabb->max;
    r->min = aabb->min;
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

extern "C" __global__ void splitNodes(AABB* aabb, Split* splits, uint depth)
{
    uint pos = (1 << (depth+1)) - 1 + 2*GlobalId;
    uint me = (1 << (depth)) - 1 + GlobalId;

    AABB node = aabb[me];
    AABB l; AABB r;

    r.min = make_float3(0,0,0);
    r.max = make_float3(0,0,0);
    l.min = make_float3(0,0,0);
    l.max = make_float3(0,0,0);
    
    Split s = splits[me];

    splitAABB(&node, s.split, s.axis, &l, &r);

    aabb[pos + 0] = l;
    aabb[pos + 1] = r;
}

extern "C" __global__ void computeSplits(AABB* aabb, uint* nodesContent, uint* nodesContentCount, Split* splits, Edge* data, uint count, uint depth,
                                         uint* scannedContentCount,
                                         uint* prefixScan,
                                         Split* nodeSplitData)
{
    uint id = GlobalId;
    
    if(id >= 2*count)
    {
        return;
    }

    uint levelOffset = elemsBeforeLevel(depth);

    uint pos = id - prefixScan[id];

    uint nodeIndex = scannedContentCount[id];

    AABB aa = aabb[levelOffset + nodeIndex];
    
    int axis = getLongestAxis(aa.min, aa.max);

    Edge edge = data[id]; //axis * 2 * count

    uint elemsInNode = nodesContentCount[levelOffset + nodeIndex];

    Split split;

    split.split = edge.getSplit(axis);
    split.axis = axis;
    split.above = elemsInNode - pos - (edge.type == eStart ? 0 : 1);
    split.below = pos + (edge.type == eEnd ? 1 : 0);
    split.primId = edge.primId;
    split.above = max((int)split.above, 0);
    split.below = max((int)split.below, 0);

    float sah = getSAH(&aa, axis, split.split, pos, elemsInNode - pos, !(split.above || split.below));

    split.sah = sah;

    splits[id] = split;
}

__device__ void rotX(float a, float3* p)
{
	float3 pt = *p;
	p->y = cos(a) * pt.y - sin(a) * pt.z;
	p->z = sin(a) * pt.y + cos(a) * pt.z;
}

__device__ void rotY(float a, float3* p)
{
	float3 pt = *p;
	p->x = cos(a) * pt.x + sin(a) * pt.z;
	p->z = -sin(a) * pt.x + cos(a) * pt.z;
}

__device__ void rotZ(float a, float3* p)
{
	float3 pt = *p;
	p->x = cos(a) * pt.x - sin(a) * pt.y;
	p->y = sin(a) * pt.x + cos(a) * pt.y;
}

__device__ void rotA(float a, float3* p, uint axis)
{
	switch(axis)
	{
		case 0 :
		{
			rotX(a, p);
		} break;
		case 1 :
		{
			rotY(a, p);
		} break;
		case 2 :
		{
			rotZ(a, p);
		} break;
	}

}

extern "C" __global__ void animateGeometry2(float* data, float time, float scale, uint parts, uint N)
{
	const uint stride = 3;

    uint id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id >= N)
    {
        return;
    }
	float t = PI_MUL2 * (float)id / (float)N;
    data[stride * id + 0] += 0.05 * cos(t + time);
    data[stride * id + 1] += 0.05 * sin(t + time);
    data[stride * id + 2] += 0.05 * cos(t + time) * cos(t + time);
}

extern "C" __global__ void animateGeometry(float* data, float time, float scale, uint parts, uint N)
{
	const uint stride = 3;

    uint id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id >= N)
    {
        return;
    }

    uint elemsPerPart = N/parts;
    elemsPerPart = elemsPerPart < 1 ? 1 : elemsPerPart;
    uint row = 4;
    uint elemsPerRow = 4 * elemsPerPart;

    uint elemntPerPartId = id % elemsPerPart;

    uint x = (id / elemsPerPart) % row;
    uint z = id / elemsPerRow;

    uint partId = id / elemsPerPart;
    
    float t = PI_MUL2 * (float)elemntPerPartId / (float)elemsPerPart;

    float dir = (partId) % 2 > 0 ? 1 : -1;

	float3 p;
	p.x = dir * cos(dir * t + time);
	p.y = dir * sin(dir * t + time) * cos(dir * t + time);
	p.z = dir * sin(dir * t + time);

	p.x = scale * p.x;
	p.y = scale * p.y;
	p.z = scale * p.z;

	rotA(5 * dir * time, &p, partId%3);

	p.x += 2 * x;
	p.y += scale;
	p.z += 2 * z;

    data[stride * id + 0] = p.x;
    data[stride * id + 1] = p.y;
    data[stride * id + 2] = p.z;
}

extern "C" __global__ void animateGeometry0(float* data, float time, float scale, uint parts, uint N)
{
    const uint stride = 3;
    uint id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id >= N)
    {
        return;
    }

    uint elemsPerPart = N/parts;
    uint row = 4;
    uint elemsPerRow = 4 * elemsPerPart;

    uint elemntPerPartId = id % elemsPerPart;

    uint x = (id / elemsPerPart) % row;
    uint z = id / elemsPerRow;

    uint partId = id / elemsPerPart;

    float t = PI_MUL2 * elemntPerPartId / (float)elemsPerPart;

    data[stride * id + 0] = 1.7*scale * x + scale * cos(t + time);
    data[stride * id + 1] = scale + scale * sin(t + time) * cos(t + time);
    data[stride * id + 2] = 1.7*scale * z + scale * sin(t + time);
}

struct vertex
{
    float3 pos;
    float3 norm;
    float2 tex;
};

__constant__ uint lpt = 24;

__device__ void addLine(vertex* lines, float3 start, float3 end, int index)
{
    uint id = threadIdx.x + blockDim.x * blockIdx.x;
    vertex v0;
    v0.pos = start;
    v0.norm = make_float3(0,1,0);
    v0.tex = make_float2(0,0);
    vertex v1;
    v1.pos = end;
    v1.tex = make_float2(1,1);
    v1.norm = make_float3(0,1,0);

    lines[lpt * id + 2*index] = v0;
    lines[lpt * id + 2*index+1] = v1;
}

extern "C" __global__ void createBBox(AABB* bbox, uint* contentCount, vertex* lines, uint N, uint d)
{
    uint stride = 8;
    uint id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id >= N)
    {
        return;
    }

    uint os = 0;//(1 << (d-1)) - 1;
    AABB bb = bbox[os + id];
    float3 m_min = bb.min;
    float3 m_max = bb.max;
    uint cc = contentCount[id];

    if(cc == 0 || abs(dot(m_min, m_max)) < 0.1)
    {
        m_max = make_float3(1,-10000,1);
        m_min = make_float3(-1,-10001,-1);
    }

    addLine(lines, m_min, make_float3(m_min.x, m_min.y, m_max.z), 0);
    addLine(lines, make_float3(m_min.x, m_min.y, m_max.z), make_float3(m_max.x, m_min.y, m_max.z), 1);
    addLine(lines, make_float3(m_max.x, m_min.y, m_max.z), make_float3(m_max.x, m_min.y, m_min.z), 2);
    addLine(lines, make_float3(m_max.x, m_min.y, m_min.z), m_min, 3);

    addLine(lines, m_min, make_float3(m_min.x, m_max.y, m_min.z), 4);
    addLine(lines, make_float3(m_min.x, m_min.y, m_max.z), make_float3(m_min.x, m_max.y, m_max.z), 5);
    addLine(lines, make_float3(m_max.x, m_min.y, m_max.z), make_float3(m_max.x, m_max.y, m_max.z), 6);
    addLine(lines, make_float3(m_max.x, m_min.y, m_min.z), make_float3(m_max.x, m_max.y, m_min.z), 7);

    addLine(lines, make_float3(m_min.x, m_max.y, m_min.z), make_float3(m_min.x, m_max.y, m_max.z), 8);
    addLine(lines, make_float3(m_min.x, m_max.y, m_max.z), make_float3(m_max.x, m_max.y, m_max.z), 9);
    addLine(lines, make_float3(m_max.x, m_max.y, m_max.z), make_float3(m_max.x, m_max.y, m_min.z), 10);
    addLine(lines, make_float3(m_max.x, m_max.y, m_min.z), make_float3(m_min.x, m_max.y, m_min.z), 11);
}
