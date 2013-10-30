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
#include "../../Nutty/Nutty/Sort.h"

#include "../Source/chimera/Logger.h"

#include <sstream>

struct SAHmin
{
    __device__ float2 operator()(float2 t0, float2 t1)
    {
        return t0.x < t1.x ? t0 : t1;
    }
};

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

template < typename T>
struct BufferPrinter
{
    uint m_c;
    uint m_r;
    BufferPrinter(uint r = (uint)-1) : m_c(0), m_r(r)
    {

    }

    void operator()(T i)
    {
        printi(i);
        m_c++;
        if(m_c % m_r == 0)
        {
            OutputDebugStringA("\n");
        }
    }
};

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

void printFloat(float i)
{
    std::stringstream ss;
    ss << i;
    ss << " ";
    OutputDebugStringA(ss.str().c_str());
}

void printFloat2(float2 i)
{
    std::stringstream ss;
    ss << "v=" << i.x;
    ss << " ";
    ss << "split=" << i.y;
    ss << ", ";
    OutputDebugStringA(ss.str().c_str());
}

void printBuffer(nutty::DeviceBuffer<uint>& b, uint r = (uint)-1)
{
    nutty::ForEach(b.Begin(), b.End(), BufferPrinter<uint>(r));
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

void printBuffer(nutty::DeviceBuffer<float>& b)
{
    nutty::ForEach(b.Begin(), b.End(), printFloat);
    OutputDebugStringA("\n");
}

void printBuffer(nutty::DeviceBuffer<float2>& b)
{
    nutty::ForEach(b.Begin(), b.End(), printFloat2);
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

struct Float3XAxisSort
{
    __device__ __host__ char operator()(float3 f0, float3 f1)
    {
        return f0.x > f1.x;
    }
};

struct Float3YAxisSort
{
    __device__ __host__ char operator()(float3 f0, float3 f1)
    {
        return f0.y > f1.y;
    }
};

struct Float3ZAxisSort
{
    __device__ __host__ char operator()(float3 f0, float3 f1)
    {
        return f0.z > f1.z;
    }
};

struct AxisSort
{
    char axis;
    AxisSort(char a) : axis(a)
    {

    }
    __device__ __host__ char operator()(float3 f0, float3 f1)
    {
        return getAxis(&f0, axis) > getAxis(&f1, axis);
    }
};

enum DeviceBufferType
{
    eData,
    eNodesContent,
    eNodesContentCount,
    eSplits,
    eSplitData,
    ePosSplits,
    eAxisAlignedBB,
    eBufferCount
};

template <
    typename T
>
class kdTree
{
private:
    void* m_pBuffer[eBufferCount];
    byte m_maxDepth;
    uint m_nodesCount;
    nutty::cuModule* m_cudaModule;
    nutty::cuKernel* m_computePossibleSplits;
    nutty::cuKernel* m_scan;
    nutty::cuKernel* m_spread;

public:
    kdTree(byte maxDepth) : m_cudaModule(NULL), m_maxDepth(maxDepth), m_nodesCount(0), m_computePossibleSplits(NULL), m_scan(NULL)
    {    
        for(int i = 0; i < eBufferCount; ++i)
        {
            m_pBuffer[i] = NULL;
        }
    }

    template <
        typename B
    >
    nutty::DeviceBuffer<B>& GetBuffer(DeviceBufferType type)
    {
        assert(m_pBuffer[type] != NULL);
        return *((nutty::DeviceBuffer<B>*)m_pBuffer[type]);
    }

    ~kdTree(void)
    {
        for(int i = 0; i < eBufferCount; ++i)
        {
            SAFE_DELETE(m_pBuffer[i]);
        }
        SAFE_DELETE(m_computePossibleSplits);
        SAFE_DELETE(m_cudaModule);
        SAFE_DELETE(m_scan);
        SAFE_DELETE(m_spread);
    }

    void Init(nutty::HostBuffer<T>& data)
    {
        nutty::DeviceBuffer<T>* d = new nutty::DeviceBuffer<T>(data.Size());
        m_pBuffer[eData] = d;

        nutty::Copy(d->Begin(), data.Begin(), data.Size());

        printBuffer(*d);

        m_nodesCount = (1 << (uint)m_maxDepth) - 1;

        InitBuffer();

        m_cudaModule = new nutty::cuModule("./ptx/KD.ptx");

        m_computePossibleSplits = new nutty::cuKernel(m_cudaModule->GetFunction("computeSplits"));
        
        m_computePossibleSplits->SetDimension(1, (uint)d->Size());
        m_computePossibleSplits->SetKernelArg(0, GetBuffer<float3>(eAxisAlignedBB));
        m_computePossibleSplits->SetKernelArg(1, GetBuffer<uint>(eNodesContent));
        m_computePossibleSplits->SetKernelArg(2, GetBuffer<uint>(eNodesContentCount));
        m_computePossibleSplits->SetKernelArg(3, GetBuffer<float2>(ePosSplits));
        m_computePossibleSplits->SetKernelArg(4, *d);

        m_scan = new nutty::cuKernel(m_cudaModule->GetFunction("scanKD"));
        m_scan->SetKernelArg(0, GetBuffer<uint>(eNodesContent));
        m_scan->SetKernelArg(1, GetBuffer<uint>(eNodesContentCount));
        m_scan->SetSharedMemory((uint)d->Size() / 2 * 2 * sizeof(uint));

        m_spread = new nutty::cuKernel(m_cudaModule->GetFunction("spreadContent"));
        m_spread->SetKernelArg(0, GetBuffer<uint>(eNodesContent));
        m_spread->SetKernelArg(1, GetBuffer<uint>(eNodesContentCount));
        m_spread->SetKernelArg(2, GetBuffer<uint>(eSplitData));
        m_spread->SetKernelArg(3, *d);
        m_spread->SetDimension(1, (uint)d->Size());
    }

    void Generate(void)
    {
        nutty::DeviceBuffer<T>& dataBuffer = GetBuffer<T>(eData);
        nutty::DeviceBuffer<float3>& aabbs = GetBuffer<float3>(eAxisAlignedBB);
        nutty::DeviceBuffer<uint>& nodesContent = GetBuffer<uint>(eNodesContent);
        nutty::DeviceBuffer<uint>& nodesContentCount = GetBuffer<uint>(eNodesContentCount);
        nutty::DeviceBuffer<float2>& posSplits = GetBuffer<float2>(ePosSplits); //SAHSplit
        nutty::DeviceBuffer<SplitData>& splitData = GetBuffer<SplitData>(eSplitData);
        
        size_t elementCount = dataBuffer.Size();

        static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
        static float3 min3f = -max3f;
        
        uint stride = (uint)elementCount;
        uint offset = stride;

        for(int i = 0; i < (m_maxDepth-1); ++i)
        {
            if(i > 0)
            {
                uint invalidAddress = (uint)-1;
                m_scan->SetDimension(1 << (1+i), (uint)elementCount / 2);
                m_scan->SetKernelArg(2, invalidAddress);
                m_scan->SetKernelArg(3, offset);
                m_scan->SetKernelArg(4, i);
                m_scan->Call();

                printBuffer(nodesContentCount);
                printBuffer(nodesContent, elementCount);

                if(i == 2)//m_maxDepth-1)
                {
                    break;
                }
            }

            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                auto start0 = aabbs.Begin() + (size_t)((2*(j-1)));
                auto start1 = aabbs.Begin() + (size_t)(2*j-1);
                
                nutty::base::ReduceIndexed(start0, dataBuffer.Begin(), dataBuffer.End(), nodesContent.Begin() + ((j-1) * elementCount), min3f, float3max());
                nutty::base::ReduceIndexed(start1, dataBuffer.Begin(), dataBuffer.End(), nodesContent.Begin() + ((j-1) * elementCount), max3f, float3min());
            }

            printBuffer(aabbs);

            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                auto start0 = aabbs.Begin() + (size_t)((2*(j-1)));
                auto start1 = aabbs.Begin() + (size_t)(2*j-1);
                float3 max = aabbs[start0];
                float3 min = aabbs[start1];
                int axis = getLongestAxis(max, min);
                SplitData sp;
                sp.axis = axis;
                splitData.Insert(j-1, sp);
                auto begin = nodesContent.Begin() + ((j-1) * elementCount);
                nutty::Sort(begin, begin + elementCount, dataBuffer.Begin(), AxisSort(axis));
            }

            printBuffer(nodesContent);

            m_computePossibleSplits->SetKernelArg(5, elementCount);
            m_computePossibleSplits->SetKernelArg(6, i);
            m_computePossibleSplits->Call();

            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                uint c = nodesContentCount[j-1];
                SplitData sp = splitData[j-1];
                nutty::base::Reduce(posSplits.Begin(), posSplits.Begin() + c, SAHmin());
                printBuffer(posSplits);
                
                float2 s = posSplits[j-1];
                sp.split = s.y;
                splitData.Insert(j-1, sp);
            }

            printBuffer(splitData);

            m_spread->SetKernelArg(4, stride);
            m_spread->SetKernelArg(5, offset);
            m_spread->SetKernelArg(6, i);
            m_spread->Call();

            printBuffer(nodesContent);
        }
    }

private:
    void InitBuffer(void)
    {
        nutty::DeviceBuffer<T>& dataBuffer = GetBuffer<T>(eData);

        size_t elementCount = dataBuffer.Size();

        float3 init = float3();
        nutty::DeviceBuffer<float3>* aabbs = new nutty::DeviceBuffer<float3>(2 * m_nodesCount, init);
        m_pBuffer[eAxisAlignedBB] = aabbs;

        nutty::DeviceBuffer<uint>* nodesContentCount = new nutty::DeviceBuffer<uint>(m_nodesCount, 0);
        m_pBuffer[eNodesContentCount] = nodesContentCount;
        nodesContentCount->Insert(0, (uint)elementCount); //first node has all content in it

        nutty::DeviceBuffer<uint>* nodesContent = new nutty::DeviceBuffer<uint>(m_nodesCount * elementCount, 0xFFFFFFFF);
        m_pBuffer[eNodesContent] = nodesContent;
        nutty::Fill(nodesContent->Begin(), nodesContent->Begin() + elementCount, nutty::unary::Sequence<uint>());

        float2 split;
        split.x = -1;
        split.y = 0;
        nutty::DeviceBuffer<float2>* possibleSplits = new nutty::DeviceBuffer<float2>(elementCount, split);
        m_pBuffer[ePosSplits] = possibleSplits;

        SplitData sinit;
        sinit.axis = (uint)-1;
        sinit.split = -0;
        nutty::DeviceBuffer<SplitData>* splitData = new nutty::DeviceBuffer<SplitData>(m_nodesCount, sinit);
        m_pBuffer[eSplitData] = splitData;
    }

    void ClearBuffer(void)
    {
        nutty::DeviceBuffer<float3>& aabbs = GetBuffer<float3>(eAxisAlignedBB);
        nutty::DeviceBuffer<uint>& nodesContent = GetBuffer<uint>(eNodesContent);
        nutty::DeviceBuffer<uint>& nodesContentCount = GetBuffer<uint>(eNodesContentCount);
    }
};

extern "C" void init(void)
{
    nutty::Init();
}

extern "C" void release(void)
{
    nutty::Release();
}

extern "C" void generate(void)
{
    uint elements = 16;
    float3 zero = float3();
    nutty::HostBuffer<float3> data(elements, zero);
    nutty::Fill(data.Begin(), data.End(), createElems());

    kdTree<float3> kd(4);

    kd.Init(data);
    kd.Generate();

    /*
    int axis = getLongestAxis(max, min);

    nutty::Sort(nodesContent.Begin(), nodesContent.Begin() + elements, content.Begin(), AxisSort(axis));
    
    nutty::cuKernel computeSplits(m.GetFunction("computeSplits"));
    nutty::DeviceBuffer<float> splits(elements);
    
    computeSplits.SetDimension(1, elements);

    computeSplits.SetKernelArg(0, aabbs);
    computeSplits.SetKernelArg(1, nodesContent);
    computeSplits.SetKernelArg(2, aabbs);
    computeSplits.SetKernelArg(3, splits);
    computeSplits.SetKernelArg(4, content);
    computeSplits.SetKernelArg(5, elements);
    computeSplits.SetKernelArg(6, d);
    computeSplits.Call();

    printBuffer(splits);
    printBuffer(nodesContent);
    printBuffer(content);

    nutty::Sort(splits.Begin(), splits.Begin() + elements, nutty::BinaryDescending<float>());

    float ss = splits[0];
    DEBUG_OUT_A("%f\n", ss);

    printBuffer(splits);

    printBuffer(nodesContentCount);
    printBuffer(nodesContent);

    SplitData initialSplit;
    initialSplit.axis = axis;
    initialSplit.split = ss;
    SplitData data; 
    data.split = 0;
    data.axis = (uint)-1;

    nutty::DeviceBuffer<SplitData> splitData(nodesCount, data);
    splitData.Insert(splitData.Begin(), initialSplit);

    printBuffer(splitData);


    //init done

    nutty::cuKernel spread(m.GetFunction("spreadContent"));
    nutty::cuKernel scanKD(m.GetFunction("_scanKD"));

    uint stride = elements;
    uint offset = stride;

    spread.SetKernelArg(0, nodesContent);
    spread.SetKernelArg(1, nodesContentCount);
    spread.SetKernelArg(2, splitData);
    spread.SetKernelArg(3, content);
    spread.SetKernelArg(4, stride);
    spread.SetKernelArg(5, offset);
    spread.SetKernelArg(6, d);
    spread.SetDimension(1, elements);
    spread.Call();

    DEBUG_OUT("\n----\n\n");
    printBuffer(nodesContent, elements);
    return;
    uint invalidAddress = (uint)-1;
    uint i = 0;
    scanKD.SetKernelArg(0, nodesContent);
    scanKD.SetKernelArg(1, nodesContentCount);
    scanKD.SetKernelArg(2, invalidAddress);
    scanKD.SetKernelArg(3, offset);
    scanKD.SetKernelArg(4, i);
    scanKD.SetDimension(1 << (1+i), elements / 2);
    scanKD.SetSharedMemory(elements / 2 * 2 * sizeof(uint));
    scanKD.Call();

    DEBUG_OUT("\n----\n\n");
    printBuffer(nodesContent, elements);

    DEBUG_OUT("\n----\n\n");
    printBuffer(nodesContentCount);

    nutty::base::ReduceIndexed(aabbs.Begin()+2, content.Begin(), content.End(), nodesContent.Begin() + ((1 << (i+1))-1) * elements, min3f, float3max());
    nutty::base::ReduceIndexed(aabbs.Begin()+3, content.Begin(), content.End(), nodesContent.Begin() + ((1 << (i+1))-1) * elements, max3f, float3min());

    nutty::base::ReduceIndexed(aabbs.Begin()+4, content.Begin(), content.End(), nodesContent.Begin() + 2*((1 << (i+1))-1) * elements, min3f, float3max());
    nutty::base::ReduceIndexed(aabbs.Begin()+5, content.Begin(), content.End(), nodesContent.Begin() + 2*((1 << (i+1))-1) * elements, max3f, float3min());

    printBuffer(aabbs);*/

}