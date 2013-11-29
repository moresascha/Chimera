#include "kdtree.cuh"
#include "../Source/chimera/Logger.h"

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
#include "../../Nutty/Nutty/Wrap.h"
#include "../../Nutty/Nutty/Scan.h"

#include "../Source/chimera/Logger.h"

#include <sstream>

template<>
struct ShrdMemory<Split>
{
    __device__ Split* Ptr(void) 
    { 
        extern __device__ __shared__ Split s_split[];
        return s_split;
    }
};

bool debug_out = true; 
 
void print(const char* str)
{
    OutputDebugStringA(str);
}

struct SAH
{
    __device__ Split operator()(Split t0, Split t1)
    {
        return t0.sah < t1.sah ? t0 : t1;
    }
};

void printi(int i)
{
    std::stringstream ss;
    ss << i;
    ss << " ";
    print(ss.str().c_str());
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
            print("\n");
        }
    }
};

void printsplit(Split i)
{
    std::stringstream ss;
    ss << "(axis=";
    ss << i.axis;
    ss << ", split=";
    ss << i.split;
    ss << ", sah=";
    ss << i.sah;
    ss << ", below=";
    ss << i.below;
    ss << ", above=";
    ss << i.above;
    ss << ")";
    print(ss.str().c_str());
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
    print(ss.str().c_str());
}

void printFloat(float i)
{
    std::stringstream ss;
    ss << i;
    ss << " ";
    print(ss.str().c_str());
}

void printFloat2(float2 i)
{
    std::stringstream ss;
    ss << "v=" << i.x;
    ss << " ";
    ss << "split=" << i.y;
    ss << ", ";
    print(ss.str().c_str());
}

void printBuffer(nutty::DeviceBuffer<uint>& b, uint r = (uint)-1)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), BufferPrinter<uint>(r));
    print("\n");
}

void printBuffer(nutty::DeviceBuffer<Split>& b)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), printsplit);
    print("\n");
}

template < typename T>
void printBuffer(T& b)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), printf3);
    print("\n");
}

void printBuffer(nutty::DeviceBuffer<float>& b)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), printFloat);
    print("\n");
}

void printBuffer(nutty::DeviceBuffer<float2>& b)
{
    if(!debug_out) return;
    nutty::ForEach(b.Begin(), b.End(), printFloat2);
    print("\n");
}

void printContentCount(nutty::DeviceBuffer<uint>& b)
{
    nutty::ForEach(b.Begin(), b.End(), printi);
    print("\n");
}

template <
    typename Data,
    typename Index
>
void printindexcontent(Data& d, Index& index, size_t size)
{
    if(!debug_out)
    {
        return;
    }
    nutty::HostBuffer<float3> t(1);
    for(int i = 0; i < size; ++i)
    {
        auto it = index + i;
        uint id = *it;
        nutty::Copy(t.Begin(), d + id, 1);
        DEBUG_OUT_A("[%f] %f %f, ", t[0].x, t[0].y, t[0].z);
    }
    DEBUG_OUT("\n");
}

template <
    typename T
>
struct clearT
{
    T operator()(void) { return T(); }
};

struct clearuint
{
    uint m_v;
    clearuint(uint v) : m_v(v)
    {

    }
    uint operator()(void) { return m_v; }
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

struct EdgeSort
{
    char axis;
    EdgeSort(char a) : axis(a)
    {

    }
    __device__ __host__ char operator()(Edge f0, Edge f1)
    {
        return f0.t > f1.t;
    }
};

__global__ void setNodesCount(uint* prefixNodesContentCount, uint* nodesContentCount, Split* allSplits, Split* nodeSplits, uint count, uint depth)
{
    uint id = GlobalId;
//     if(id >= (1 << depth))
//     {
//         return;
//     }
    uint contentCountOffset = elemsBeforeNextLevel(depth);//(1 << (depth+1)) - 1;
	uint contentCountOffsetMe = elemsBeforeLevel(depth); //(1 << depth) - 1;
    uint sa = 0;
    int last = -1;
    for(uint i = 0; i < id; ++i)
    {
        last = nodesContentCount[contentCountOffsetMe + i];
        sa += last;
    }

    Split s = allSplits[sa];
    s.contentStartIndex = sa;
    if(last > 0 || sa == 0)
    {
        nodesContentCount[contentCountOffset + 2 * id + 0] = s.below;
        nodesContentCount[contentCountOffset + 2 * id + 1] = s.above;
    }
 	nodeSplits[contentCountOffsetMe + id] = s;
}

struct SphereBBox
{
    __device__ AABB operator()(float3 pos)
    {
        AABB bbox;

        bbox.min = pos - make_float3(0.5, 0.5, 0.5);
        bbox.max = pos + make_float3(0.5, 0.5, 0.5);

        return bbox;
    }
};

template <
    typename T,
    typename BBoxComputer
>
__global__ void computePerPrimBBox(T* prims, AABB* aabbs, BBoxComputer bboxc, uint N)
{
    uint id = GlobalId;
    if(id >= N)
    {
        return;
    }
    aabbs[id] = bboxc(prims[id]);
}

__global__ void computePerPrimEdges(Edge* edges, AABB* aabbs, uint N)
{
    uint id = GlobalId;
    if(id >= N)
    {
        return;
    }

    AABB aabb = aabbs[id];

    Edge start;
    start.type = eStart;
    Edge end;
    end.type = eEnd;

    for(byte i = 0; i < 3; ++i)
    {
        start.t = getAxis(&aabb.min, i);
        end.t = getAxis(&aabb.max, i);
        edges[2 * N * i + 2 * id + 0] = start;
        edges[2 * N * i + 2 * id + 1] = end;
    }
}



template <
    typename T
>
class kdTree : public IKDTree
{
public:
    void* m_pBuffer[eBufferCount];
    void* m_pClearBuffer[eBufferCount];
    byte m_depth;
    byte m_maxDepth;
    uint m_nodesCount;
    uint m_elements;
    nutty::DeviceBuffer<float3>* m_data;
    nutty::cuModule* m_cudaModule;
    nutty::cuKernel* m_computePossibleSplits;
    nutty::cuKernel* m_scan;
    nutty::cuKernel* m_spread;
    nutty::cuKernel* m_splitBBox;

    nutty::HostBuffer<Split> splitTmp;
    nutty::HostBuffer<Split> splitDataTmp;
    nutty::HostBuffer<uint> hostContentCount;
    nutty::HostBuffer<uint> countTmp;
    nutty::HostBuffer<float3> bboxTmp;

    nutty::DeviceBuffer<uint> m_perThreadNodePos;
	nutty::DeviceBuffer<uint> m_prefixSumNodeCount;

    nutty::DeviceBuffer<float3> m_aabbMin;
    nutty::DeviceBuffer<float3> m_aabbMax;

    nutty::DeviceBuffer<AABB> m_primBBox;
    nutty::DeviceBuffer<Edge> m_edges;

    nutty::DeviceBuffer<uint> m_null;

    nutty::cuStreamPool* m_pStreamPool;

    nutty::cuStream m_defaultStream;

    kdTree(byte depth, byte maxDepth) : m_cudaModule(NULL), m_depth(depth), m_maxDepth(maxDepth), m_nodesCount(0), m_computePossibleSplits(NULL), m_scan(NULL), m_elements(0)
    {    
        for(int i = 0; i < eBufferCount; ++i)
        {
            m_pBuffer[i] = m_pClearBuffer[i] = NULL;
        }
        countTmp.Resize(1 << maxDepth);
        bboxTmp.Resize(1 << maxDepth);
        m_pStreamPool = new nutty::cuStreamPool();
    }

    std::stringstream _stream;
    void GetContentCountStr(std::string& s)
    {
        nutty::Copy(hostContentCount.Begin(), GetBuffer<uint>(eNodesContentCount).Begin(), m_nodesCount);
        s.clear();
        _stream.str("");
        for(uint i = 0; i < (1<<m_depth)-1; ++i)
        {
            _stream << hostContentCount[i] << " ";
        }

        s = _stream.str();
    }

    template <
        typename B
    >
    nutty::DeviceBuffer<B>& GetBuffer(DeviceBufferType type)
    {
        assert(m_pBuffer[type] != NULL);
        return *((nutty::DeviceBuffer<B>*)m_pBuffer[type]);
    }
    
    uint GetCurrentDepth(void)
    {
        return m_depth;
    }

    void* GetData(void)
    {
        return m_data;
    }

    void SetDepth(uint d)
    {
        m_depth = min(m_maxDepth, max(1, d));
    }

    ~kdTree(void)
    {
        for(int i = 0; i < eBufferCount; ++i)
        {
            SAFE_DELETE(m_pBuffer[i]);
            SAFE_DELETE(m_pClearBuffer[i]);
        }
        SAFE_DELETE(m_data);
        SAFE_DELETE(m_computePossibleSplits);
        SAFE_DELETE(m_cudaModule);
        SAFE_DELETE(m_scan);
        SAFE_DELETE(m_spread);
        SAFE_DELETE(m_splitBBox);
        SAFE_DELETE(m_pStreamPool);
    }

    void Init(void* data, uint elements)
    {
        m_data = (nutty::DeviceBuffer<T>*)(data); //(nutty::MappedBufferPtr<T>*)data;

        m_primBBox.Resize(elements);
        m_edges.Resize(elements * 3 * 2); //3 axis, 2 edges per element

        m_elements = elements;

        m_nodesCount = (1 << (uint)m_maxDepth) - 1;

        splitTmp.Resize(elements);

        hostContentCount.Resize(m_nodesCount);
        
        splitDataTmp.Resize((m_nodesCount+1)/2);

        m_aabbMin.Resize(m_elements);
        m_aabbMax.Resize(m_elements);

        InitBuffer();

        m_cudaModule = new nutty::cuModule("./ptx/KD.ptx");

        m_computePossibleSplits = new nutty::cuKernel(m_cudaModule->GetFunction("computeSplits"));
        uint block = elements < 256 ? elements : 256;
        uint grid = max(1, (uint)elements / block);
        m_computePossibleSplits->SetDimension(grid, block);
        m_computePossibleSplits->SetKernelArg(0, GetBuffer<float3>(eAxisAlignedBB));
        m_computePossibleSplits->SetKernelArg(1, GetBuffer<uint>(eNodesContent));
        m_computePossibleSplits->SetKernelArg(2, GetBuffer<uint>(eNodesContentCount));
        m_computePossibleSplits->SetKernelArg(3, GetBuffer<float2>(ePosSplits));
        m_computePossibleSplits->SetKernelArg(7, m_perThreadNodePos);
        m_computePossibleSplits->SetKernelArg(8, m_prefixSumNodeCount);
        m_computePossibleSplits->SetKernelArg(9, GetBuffer<uint>(eSplitData));
            

        m_scan = new nutty::cuKernel(m_cudaModule->GetFunction("scanKD"));
        m_scan->SetKernelArg(0, GetBuffer<uint>(eNodesContent));
        m_scan->SetKernelArg(1, GetBuffer<uint>(eNodesContentCount));
        m_scan->SetSharedMemory((uint)elements * 2 * sizeof(uint));

        m_spread = new nutty::cuKernel(m_cudaModule->GetFunction("spreadContent"));
        m_spread->SetKernelArg(0, GetBuffer<uint>(eNodesContent));
        m_spread->SetKernelArg(1, GetBuffer<uint>(eNodesContentCount));
        m_spread->SetKernelArg(2, GetBuffer<uint>(eSplitData));
        m_spread->SetDimension(grid, block);

        m_splitBBox = new nutty::cuKernel(m_cudaModule->GetFunction("splitNodes"));

        m_splitBBox->SetKernelArg(0, GetBuffer<uint>(eAxisAlignedBB));
        m_splitBBox->SetKernelArg(1, GetBuffer<uint>(eSplitData));

        computePerPrimBBox<<<grid, block>>>(m_data->Begin()(), m_primBBox.Begin()(), SphereBBox(), elements);
        computePerPrimEdges<<<grid, block>>>(m_edges.Begin()(), m_primBBox.Begin()(), elements);
    }

    void Generate(void)
    {
        nutty::DeviceBuffer<float3>& aabbs = GetBuffer<float3>(eAxisAlignedBB);
        nutty::DeviceBuffer<uint>& nodesContent = GetBuffer<uint>(eNodesContent);
        nutty::DeviceBuffer<uint>& nodesContentCount = GetBuffer<uint>(eNodesContentCount);
        nutty::DeviceBuffer<Split>& posSplits = GetBuffer<Split>(ePosSplits); //SAHSplit
        nutty::DeviceBuffer<Split>& splitData = GetBuffer<Split>(eSplitData);
        
        size_t elementCount = m_elements;

        static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
        static float3 min3f = -max3f;
        
        uint stride = (uint)elementCount;

        auto dataBuffer = m_data->Begin();

        m_computePossibleSplits->SetKernelArg(4, *m_data);
        m_spread->SetKernelArg(3, *m_data);

        const nutty::cuStream& smin = m_pStreamPool->PeekNextStream();
        nutty::SetStream(smin);
        nutty::Reduce(m_aabbMin.Begin(), dataBuffer, dataBuffer + elementCount, float3min());
        m_defaultStream.WaitEvent(std::move(smin.RecordEvent()));

        const nutty::cuStream& smax = m_pStreamPool->PeekNextStream();
        nutty::SetStream(smax);
        nutty::Reduce(m_aabbMax.Begin(), dataBuffer, dataBuffer + elementCount, float3max());
        m_defaultStream.WaitEvent(std::move(smax.RecordEvent()));

        nutty::Copy(aabbs.Begin(), m_aabbMin.Begin(), 1);
        nutty::Copy(aabbs.Begin()+1, m_aabbMax.Begin(), 1);

        for(int i = 0; i < m_depth-1; ++i)
        {           
            uint contentSum = 0;
	
            uint copyStartAdd = (1 << i) - 1;
            uint copyLength = ((1 << (i+1)) - 1) - copyStartAdd;

            nutty::Copy(bboxTmp.Begin() + 2 * copyStartAdd, aabbs.Begin() + 2 * copyStartAdd, copyLength * 2);
			
            nutty::Copy(countTmp.Begin() + copyStartAdd, nodesContentCount.Begin() + copyStartAdd, copyLength);

            //printBuffer(nodesContent);
            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                float3 min = bboxTmp[(size_t)(2*(j-1))];
                float3 max = bboxTmp[(size_t)(2*j-1)];
                
                int axis = getLongestAxis(min, max);

                //auto begin = nodesContent.Begin() + contentSum;
                auto begin = dataBuffer + contentSum;

                uint c = countTmp[(size_t)(j-1)];
                if(c > 0)
                {                
                    //nutty::Sort(begin, begin + c, dataBuffer, AxisSort(axis)); key/value
                    const nutty::cuStream& s = m_pStreamPool->PeekNextStream();
                    nutty::SetStream(s);
                    nutty::Sort(begin, begin + c, AxisSort(axis));
                    m_defaultStream.WaitEvent(std::move(s.RecordEvent()));
                    //printindexcontent(dataBuffer, nodesContent.Begin(), m_elements);
                }
                contentSum += c;
            }

           // printBuffer(nodesContent);
            nutty::SetStream(m_defaultStream);
            
            m_computePossibleSplits->SetKernelArg(5, elementCount);
            m_computePossibleSplits->SetKernelArg(6, i);
            m_computePossibleSplits->Call();

            m_defaultStream.ClearEvents();

            contentSum = 0;

            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                int c = (int)countTmp[(size_t)(j-1)];
                if((int)(c) > 0)
                {
                    auto begin = posSplits.Begin() + contentSum;
                    if(c > 1)
                    {
                        const nutty::cuStream& s = m_pStreamPool->PeekNextStream();
                        nutty::SetStream(s);
                        nutty::Reduce(begin, begin + c, SAH());
                        m_defaultStream.WaitEvent(std::move(s.RecordEvent()));
                    }

//                     printsplit(*begin);
//                     DEBUG_OUT("\n"); 
                }
                //if((int)(c) > 0)
                {
                    contentSum += c;
                }
            }

            uint g = 1;
            uint b = (1 << i);

            if(b > 256)
            {
                g = nutty::cuda::getCudaGrid(b, 256U);
                b = 256;
            }

            nutty::SetStream(m_defaultStream);

            setNodesCount<<<g, b>>>(m_prefixSumNodeCount.Begin()(), nodesContentCount.Begin()(), posSplits.Begin()(), splitData.Begin()(), m_elements, i);

            m_defaultStream.ClearEvents();

//             if(i+1 == m_depth-1)
//             {
//                 break;
//             }

            m_spread->SetKernelArg(4, stride);
            m_spread->SetKernelArg(5, i);
            m_spread->SetKernelArg(6, m_perThreadNodePos);
            m_spread->SetKernelArg(7, m_prefixSumNodeCount);
            m_spread->Call();

            m_splitBBox->SetDimension(g, b);
            m_splitBBox->SetKernelArg(2, i);
            m_splitBBox->Call();
        }
    }

    void Update(void)
    {
        ClearBuffer();
        Generate();
    }

    void* GetBuffer(DeviceBufferType b)
    {
        return m_pBuffer[b];
    }

private:
    void InitBuffer(void)
    {
        size_t elementCount = m_elements;

        float3 init = float3();
        nutty::DeviceBuffer<float3>* aabbs = new nutty::DeviceBuffer<float3>(2 * m_nodesCount, init);
        m_pBuffer[eAxisAlignedBB] = aabbs;
        m_pClearBuffer[eAxisAlignedBB] = new nutty::DeviceBuffer<float3>(2 * m_nodesCount, init);

        nutty::DeviceBuffer<uint>* nodesContentCount = new nutty::DeviceBuffer<uint>(m_nodesCount, 0);
        m_pBuffer[eNodesContentCount] = nodesContentCount;
        nodesContentCount->Insert(0, (uint)elementCount); //first node has all content in it

        nodesContentCount = new nutty::DeviceBuffer<uint>(m_nodesCount, 0);
        nodesContentCount->Insert(0, (uint)elementCount); //first node has all content in it
        m_pClearBuffer[eNodesContentCount] = nodesContentCount;

        m_prefixSumNodeCount.Resize(elementCount);
        m_perThreadNodePos.Resize(elementCount);
        m_null.Resize(elementCount);

        nutty::HostBuffer<uint> null(m_prefixSumNodeCount.Size(), 0);

        nutty::Copy(m_null.Begin(), null.Begin(), null.Size());
        nutty::Copy(m_prefixSumNodeCount.Begin(), m_null.Begin(), m_prefixSumNodeCount.Size());
        nutty::Copy(m_perThreadNodePos.Begin(), m_null.Begin(), m_perThreadNodePos.Size());

        nutty::DeviceBuffer<uint>* nodesContent = new nutty::DeviceBuffer<uint>(elementCount, 0xFFFFFFFF);
        m_pBuffer[eNodesContent] = nodesContent;
        nutty::Fill(nodesContent->Begin(), nodesContent->Begin() + elementCount, nutty::unary::Sequence<uint>());

        nodesContent = new nutty::DeviceBuffer<uint>(elementCount, 0xFFFFFFFF);
        m_pClearBuffer[eNodesContent] = nodesContent;
        nutty::Fill(nodesContent->Begin(), nodesContent->Begin() + elementCount, nutty::unary::Sequence<uint>());

        Split s;
        ZeroMemory(&s, sizeof(Split)); 
        s.sah = FLT_MAX;

        nutty::DeviceBuffer<Split>* possibleSplits = new nutty::DeviceBuffer<Split>(elementCount, s);
        m_pBuffer[ePosSplits] = possibleSplits;

        possibleSplits = new nutty::DeviceBuffer<Split>(elementCount, s);
        m_pClearBuffer[ePosSplits] = possibleSplits;

        nutty::DeviceBuffer<Split>* splitData = new nutty::DeviceBuffer<Split>((m_nodesCount+1)/2, s);
        m_pBuffer[eSplitData] = splitData;

        splitData = new nutty::DeviceBuffer<Split>((m_nodesCount+1)/2, s);
        m_pClearBuffer[eSplitData] = splitData;
    }

public:
    void ClearBuffer(void)
    {
        nutty::DeviceBuffer<float3>& aabbs = GetBuffer<float3>(eAxisAlignedBB);
        nutty::DeviceBuffer<float3>* _clearaabbs = (nutty::DeviceBuffer<float3>*)m_pClearBuffer[eAxisAlignedBB];
        nutty::Copy(aabbs.Begin(), _clearaabbs->Begin(), aabbs.Size());

        nutty::DeviceBuffer<uint>& nodesContent = GetBuffer<uint>(eNodesContent);
        nutty::DeviceBuffer<uint>* _clearNodes = (nutty::DeviceBuffer<uint>*)m_pClearBuffer[eNodesContent];
        nutty::Copy(nodesContent.Begin(), _clearNodes->Begin(), nodesContent.Size());

        nutty::DeviceBuffer<uint>& nodesContentCount = GetBuffer<uint>(eNodesContentCount);
        nutty::DeviceBuffer<uint>* _clearNodesC = (nutty::DeviceBuffer<uint>*)m_pClearBuffer[eNodesContentCount];
        nutty::Copy(nodesContentCount.Begin(), _clearNodesC->Begin(), nodesContentCount.Size());

        nutty::DeviceBuffer<Split>& splits = GetBuffer<Split>(eSplitData);
        nutty::DeviceBuffer<Split>* _clearSplit = (nutty::DeviceBuffer<Split>*)m_pClearBuffer[eSplitData];
        nutty::Copy(splits.Begin(), _clearSplit->Begin(), splits.Size());

        nutty::DeviceBuffer<Split>* _clearPosSplit = (nutty::DeviceBuffer<Split>*)m_pClearBuffer[ePosSplits];
        nutty::DeviceBuffer<Split>& possibleSplits = GetBuffer<Split>(ePosSplits);
        nutty::Copy(possibleSplits.Begin(), _clearPosSplit->Begin(), possibleSplits.Size());

        nutty::Copy(m_prefixSumNodeCount.Begin(), m_null.Begin(), m_prefixSumNodeCount.Size());
        nutty::Copy(m_perThreadNodePos.Begin(), m_null.Begin(), m_perThreadNodePos.Size());
    }
};

std::vector<kdTree<float3>*> g_trees;

extern "C" void init(void)
{
    nutty::Init();
}

extern "C" void release(void)
{
    for(auto it = g_trees.begin(); it != g_trees.end(); ++it)
    {
        delete *it;
    }
    nutty::Release();
}

extern "C" IKDTree* generate(nutty::DeviceBuffer<float3>* data, uint elements, uint d, uint maxDepth)
{
    kdTree<float3>* tree = new kdTree<float3>(d, maxDepth);
    tree->Init((void*)data, elements);
    tree->Generate();
    g_trees.push_back(tree);
    return tree;
}
