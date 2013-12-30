#include "kdtree.cuh"
#include <math_functions.h>

#define EXTC extern "C"

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

EXTC __global__ void animateGeometry2(float* data, float time, float scale, uint parts, uint N)
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

EXTC __global__ void animateGeometry(float* data, float time, float scale, uint parts, uint N)
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

EXTC __global__ void animateGeometry0(float* data, float time, float scale, uint parts, uint N)
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

    //uint partId = id / elemsPerPart;

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

    if(isnan(start.x) || isnan(start.y) || isnan(start.z) || isnan(end.x) || isnan(end.y) || isnan(end.z))
    {
        start.x = start.y = start.z = -1;
        end.x = end.y = end.z = -2;
    }

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


EXTC __global__ void createBBox(BBox* bbox, Node nodes, vertex* lines, uint N, uint d)
{
    //uint stride = 8;
    uint id = threadIdx.x + blockDim.x * blockIdx.x;  

    if(id >= N)
    {
        return;
    }

    uint os = 0;//(1 << (d-1)) - 1;
    BBox bb = bbox[os + id];
    float3 m_min = bb.min;
    float3 m_max = bb.max;
    uint cc = nodes.contentCount[id];

    if(cc == 0 || abs(dot(m_min, m_max)) < 0.1)
    {
        m_max = make_float3(1,-10000, 1);
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