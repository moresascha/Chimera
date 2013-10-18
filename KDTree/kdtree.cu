#include "kdtree.cuh"
#include <windows.h>
#include "../../Nutty/Nutty/Nutty.h"
#include "../../Nutty/Nutty/Fill.h"
#include "../../Nutty/Nutty/ForEach.h"
#include "../../Nutty/Nutty/Reduce.h"
#include "../../Nutty/Nutty/Functions.h"
#include "../../Nutty/Nutty/DeviceBuffer.h"
#include "../../Nutty/Nutty/cuda/Module.h"
#include "../../Nutty/Nutty/cuda/Kernel.h"

#include "../Source/chimera/Logger.h"

#include <sstream>

struct float3min
{
    __device__ float3 operator()(float3 t0, float3 t1)
    {
        float3 r;
        r.x = nutty::binary::Min<float>()(t0.x, t1.x);
        r.y = nutty::binary::Min<float>()(t0.y, t1.y);
        r.z = nutty::binary::Min<float>()(t0.z, t1.z);
        return r;
    }
};

struct float3max
{
    __device__  float3 operator()(float3 t0, float3 t1)
    {
        float3 r;
        r.x = nutty::binary::Max<float>()(t0.x, t1.x);
        r.y = nutty::binary::Max<float>()(t0.y, t1.y);
        r.z = nutty::binary::Max<float>()(t0.z, t1.z);
        return r;
    }
};

void printi(int i)
{
    std::stringstream ss;
    ss << i;
    ss << " ";
    OutputDebugStringA(ss.str().c_str());
}

void printsplit(SplitData i)
{
    std::stringstream ss;
    ss << "(axis=";
    ss << i.axis;
    ss << ", ";
    ss << "split=";
    ss << i.split;
    ss << ") ";
    OutputDebugStringA(ss.str().c_str());
}

void printf3(float3 i)
{
    std::stringstream ss;
    ss << "(x=";
    ss << i.x;
    ss << ", y=";
    ss << i.y;
    ss << ", z=";
    ss << i.z;
    ss << ") ";
    OutputDebugStringA(ss.str().c_str());
}

void printBuffer(nutty::DeviceBuffer<uint>& b)
{
    nutty::ForEach(b.Begin(), b.End(), printi);
    OutputDebugStringA("\n");
}

void printBuffer(nutty::DeviceBuffer<SplitData>& b)
{
    nutty::ForEach(b.Begin(), b.End(), printsplit);
    OutputDebugStringA("\n");
}

void printBuffer(nutty::DeviceBuffer<float3>& b)
{
    nutty::ForEach(b.Begin(), b.End(), printf3);
    OutputDebugStringA("\n");
}

struct createElems
{
    float3 operator()(void)
    {
        float3 f;
        f.x = -1.0f + 2.0f * rand() / (float)RAND_MAX;
        f.y = -1.0f + 2.0f * rand() / (float)RAND_MAX;
        f.z = -1.0f + 2.0f * rand() / (float)RAND_MAX;
        return f;
    }
};

extern "C" void generate(void)
{
    nutty::Init();

    nutty::cuModule m("./ptx/KD.cu");

    nutty::cuKernel spread(m.GetFunction("spreadContent"));
    nutty::cuKernel scaneKD(m.GetFunction("_scanKD"));
    nutty::cuKernel computeSplits(m.GetFunction("computeSplits"));

    uint elements = 16;

    nutty::DeviceBuffer<float3> content(elements);

    nutty::Fill(content.Begin(), content.End(), createElems());

    printBuffer(content);

    float3 min = nutty::Reduce(float3min(), content);
    float3 max = nutty::Reduce(float3max(), content);

    float3 v = max - min;
    DEBUG_OUT_A("%f %f %f\n", v.x, v.y, v.z); 
    DEBUG_OUT_A("%f %f %f\n", min.x, min.y, min.z); 
    DEBUG_OUT_A("%f %f %f\n", max.x, max.y, max.z);

    SplitData initialSplit;
    
    getSplit(min, max, &initialSplit.split, &initialSplit.axis);

    uint treeDepth = 4;
    uint nodesCount = (1 << treeDepth) - 1;
    
    nutty::DeviceBuffer<uint> nodesContentCount(nodesCount, 0);
    nodesContentCount.Insert(nodesContentCount.Begin(), nodesCount); //first node has all content in it

    printBuffer(nodesContentCount);

    nutty::DeviceBuffer<uint> nodesContent(nodesCount * elements, 0xFFFFFFFF);

    nutty::Fill(nodesContent.Begin(), nodesContent.Begin() + elements, nutty::unary::Sequence<uint>());

    printBuffer(nodesContent);

    SplitData data;
    data.split = 0;
    data.axis = -1;

    nutty::DeviceBuffer<SplitData> splitData(nodesCount, data);
    splitData.Insert(splitData.Begin(), initialSplit);

    printBuffer(splitData);

    nutty::Release();
}