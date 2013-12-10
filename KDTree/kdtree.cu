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

#include "cuTimer.cuh"

#include "kd_kernel.cuh"

#include <sstream>

bool g_treeDebug = false; 
 
void print(const char* str)
{
    OutputDebugStringA(str);
}

void printaabb(AABB aabb)
{
    std::stringstream ss; 
    ss << "(" << aabb.min.x << ", " << aabb.min.y << ", " << aabb.min.z;
    ss << " | " << aabb.max.x << ", " << aabb.max.y << ", " << aabb.max.z << ")";
    print(ss.str().c_str());
}

void printf3(float3 data)
{
    std::stringstream ss; 
    ss << "(" << data.x << ", " << data.y << ", " << data.z << ") ";
    print(ss.str().c_str());
}


void printSplit(Split&b, uint i)
{
    if(!g_treeDebug) return;
    nutty::DevicePtr<uint> below(b.below + i);
    nutty::DevicePtr<uint> above(b.above + i);
    nutty::DevicePtr<byte> axis(b.axis + i);
    nutty::DevicePtr<uint> primId(b.primId + i);
    nutty::DevicePtr<IndexedSAH> isah(b.sah + i);
    nutty::DevicePtr<float> split(b.split + i);

    std::stringstream ss;
    ss << "(index=" << 
    i << ", axis=" << 
    (int)axis[0] << ", split=" << 
    split[0] << ", sah=" << 
    isah[0].sah << ", primId=" <<   
    primId[0] << ", below=" << 
    below[0] << ", above=" << 
    above[0] << ")\n";
    print(ss.str().c_str());
}

void printEdge(Edge&b, uint i)
{
    if(!g_treeDebug) return;
    nutty::DevicePtr<Indexed3DEdge> ie(b.indexedEdge + i);
    nutty::DevicePtr<uint> primId(b.primId + i);
    nutty::DevicePtr<EdgeType> type(b.type + i);

    std::stringstream ss;
    ss << "(index=" << 
    ie[0].index << ", x=" << ie[0].t3.x <<
    i << ", y=" << ie[0].t3.y <<
    i << ", z=" << ie[0].t3.z <<
    ", primId=" << primId[0] <<
    ", type=" << type[0] << ") ";
    print(ss.str().c_str());
}

void printSplits(Split& b, uint size)
{
    if(!g_treeDebug) return;

    print("Splits:\n");
    for(uint i = 0; i < size; ++i)
    {
        printSplit(b, i);
    }

    print("\n");
}

void printBuffer(Edge& b, uint size)
{
    if(!g_treeDebug) return;

    print("Edges:\n");
    for(uint i = 0; i < size; ++i)
    {
        printEdge(b, i);
    }

    print("\n");
}

void printBuffer(Node& b, uint size)
{
    if(!g_treeDebug) return;

    nutty::DevicePtr<byte> axis(b.axis);
    nutty::DevicePtr<uint> cc(b.contentCount);
    nutty::DevicePtr<uint> start(b.contentStartIndex);
    nutty::DevicePtr<byte> isLeaf(b.leaf);
    nutty::DevicePtr<float> split(b.split);
    
    std::stringstream ss;
    ss << "Nodes: ";
    for(uint i = 0; i < size; ++i)
    {
        ss << "(index=" << i << ", axis=" << (int)axis[i] << ", split=" << split[i] << ", leaf=" << (int)isLeaf[i] << ", start=" << start[i] << ", count=" << cc[i] << ")";
    }
    print(ss.str().c_str());
    print("\n");
}

void printBuffer(nutty::DeviceBuffer<AABB>& b)
{
    if(!g_treeDebug) return;
    nutty::ForEach(b.Begin(), b.End(), printaabb);
    print("\n");
}

void printBuffer(nutty::DeviceBuffer<float3>& b)
{
    if(!g_treeDebug) return;
    nutty::ForEach(b.Begin(), b.End(), printf3);
    print("\n");
}

std::stringstream & operator << (std::stringstream &out, Indexed3DEdge const &t)
{
    out << t.index << " " << t.t3.x << " ";
    return out;
}

template <
    typename T
>
void printT(T t)
{
    std::stringstream ss;
    ss << t << " ";
    print(ss.str().c_str());
}


template <
    typename T
>
void printBuffer(nutty::DeviceBuffer<T>& b)
{
    if(!g_treeDebug) return;
    nutty::ForEach(b.Begin(), b.End(), printT<T>);
    print("\n");
} 

//--------print stuff end

#define CLEAR_BUFFER(__buffer) nutty::Copy(__buffer.Begin(), m_null.Begin(), __buffer.Size())

template<>
struct ShrdMemory<IndexedSAH>
{
    __device__ IndexedSAH* Ptr(void) 
    { 
        extern __device__ __shared__ IndexedSAH s_split[];
        return s_split;
    }
};

template<>
struct ShrdMemory<Indexed3DEdge>
{
    __device__ Indexed3DEdge* Ptr(void) 
    { 
        extern __device__ __shared__ Indexed3DEdge s_edge[];
        return s_edge;
    }
};


struct SAH
{
    __device__ Split operator()(Split t0, Split t1)
    {
        return t0.sah < t1.sah ? t0 : t1;
    }
};

struct ReduceIndexedSAH
{
    __device__ __host__ IndexedSAH operator()(IndexedSAH t0, IndexedSAH t1)
    {
        return t0.sah < t1.sah ? t0 : t1;
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

struct EdgeSort
{
    byte axis;
    EdgeSort(byte a) : axis(a)
    {

    }
    __device__ __host__ char operator()(Indexed3DEdge f0, Indexed3DEdge f1)
    {
        return getAxis(&f0.t3, axis) > getAxis(&f1.t3, axis);
    }
};

template <
    typename T
>
class kdTree : public IKDTree
{
public:
    byte m_depth;
    byte m_maxDepth;
    uint m_nodesCount;
    uint m_elements;

    nutty::DeviceBuffer<T>* m_data;
    nutty::DeviceBuffer<T> m_transformedData;

    nutty::cuModule* m_cudaModule;
    nutty::cuKernel* m_computePossibleSplits;
    nutty::cuKernel* m_spread;
    nutty::cuKernel* m_splitBBox;

    nutty::HostBuffer<uint> hostContentCount;
    nutty::HostBuffer<uint> countTmp;
    nutty::HostBuffer<AABB> bboxTmp;

    nutty::DeviceBuffer<uint> m_perThreadNodePos;
	nutty::DeviceBuffer<uint> m_prefixSumNodeCount;
    
    nutty::DeviceBuffer<byte> m_nodeSplitAxis;
    nutty::DeviceBuffer<byte> m_nodeIsLeaf;
    nutty::DeviceBuffer<uint> m_nodeContentCount;
    nutty::DeviceBuffer<uint> m_nodeContentStartIndex;
    nutty::DeviceBuffer<float> m_nodeSplit;
    nutty::DeviceBuffer<uint> m_nodeBelow;
    nutty::DeviceBuffer<Node> m_nodesBuffer;
    Node m_nodeStructOfArrayArg;

    nutty::DeviceBuffer<byte> m_splitAxis[2];
    nutty::DeviceBuffer<uint> m_splitPrimId[2];
    nutty::DeviceBuffer<float> m_splitSplitT[2];
    nutty::DeviceBuffer<IndexedSAH> m_splitSAH;
    nutty::DeviceBuffer<uint> m_splitBelow[2];
    nutty::DeviceBuffer<uint> m_splitAbove[2];
    Split m_splitStructOfArrayArg[2];

    nutty::DeviceBuffer<EdgeType> m_edgeType[2];
    nutty::DeviceBuffer<uint> m_edgePrimId[2];
    nutty::DeviceBuffer<Indexed3DEdge> m_edgeIndexedEdge;
    /*nutty::DeviceBuffer<float> m_edgeTy;
    nutty::DeviceBuffer<float> m_edgeTz;*/
    Edge m_edgeStructOfArrayArg[2];

    nutty::DeviceBuffer<float3> m_aabbMin;
    nutty::DeviceBuffer<float3> m_aabbMax;

    nutty::DeviceBuffer<AABB> m_nodeAABBs;

    nutty::DeviceBuffer<AABB> m_primBBox;  

    nutty::DeviceBuffer<uint> m_null;

    nutty::cuStreamPool* m_pStreamPool;

    nutty::cuStream m_defaultStream;

    byte m_toggleBit;

    cuTimer m_timer[16];

    kdTree(byte depth, byte maxDepth) 
        : m_cudaModule(NULL), 
        m_depth(depth), 
        m_maxDepth(maxDepth), 
        m_nodesCount(0), 
        m_computePossibleSplits(NULL), 
        m_elements(0),
        m_toggleBit(0)
    {    
        assert(maxDepth >= depth);
        m_pStreamPool = new nutty::cuStreamPool();
    }

    std::stringstream _stream;
    void GetContentCountStr(std::string& s)
    {
        uint limit = 32;
        nutty::Copy(hostContentCount.Begin(), m_nodeContentCount.Begin(), min(m_nodesCount, limit));
        s.clear();
        _stream.str("");
        _stream << "Prims:" << m_elements << " - ";
        for(uint i = 0; i < min(m_nodesCount, limit); ++i)
        {
            _stream << hostContentCount[i] << " ";
        }

        s = _stream.str();
    }

    uint GetCurrentDepth(void)
    {
        return m_depth;
    }

    void* GetData(void)
    {
        return &m_transformedData;
    }

    nutty::DeviceBuffer<Node>* GetNodes(void)
    {
        return &m_nodesBuffer;
    }

    nutty::cuStream& GetDefaultStream(void)
    {
        return m_defaultStream;
    }


    nutty::DeviceBuffer<AABB>* GetAABBs(void)
    {
        return &m_nodeAABBs;
    }

    void SetDepth(uint d)
    {
        m_depth = min(m_maxDepth, max(2, d));
    }

    ~kdTree(void)
    {
        SAFE_DELETE(m_data);
        SAFE_DELETE(m_pStreamPool);
    }

    void Init(void* data, uint elements)
    {
        m_elements = elements;

        m_nodesCount = (1 << (uint)m_maxDepth) - 1;

        m_data = (nutty::DeviceBuffer<T>*)(data);
        
        InitBuffer();
    }

    void Generate(void)
    {
        static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
        static float3 min3f = -max3f;

        m_toggleBit = 0;
        
        auto dataBuffer = m_data->Begin();

        uint _block = m_elements < 256 ? m_elements : 256;
        uint _grid = nutty::cuda::GetCudaGrid(m_elements, _block);

        nutty::cuStream& sbbox = m_pStreamPool->PeekNextStream();
        nutty::cuStream& sedges = m_pStreamPool->PeekNextStream();
        computePerPrimBBox<<<_grid, _block, 0, sbbox()>>>
            (m_data->Begin()(), m_primBBox.Begin()(), SphereBBox(), m_elements);
        sedges.WaitEvent(std::move(sbbox.RecordEvent()));

        computePerPrimEdges<<<_grid, _block, 0, sedges()>>>
            (m_edgeStructOfArrayArg[m_toggleBit], m_primBBox.Begin()(), m_elements);
        m_defaultStream.WaitEvent(std::move(sedges.RecordEvent()));

        const nutty::cuStream& smin = m_pStreamPool->PeekNextStream();
        nutty::SetStream(smin);
        nutty::Reduce(m_aabbMin.Begin(), dataBuffer, dataBuffer + m_elements, float3min());
        m_defaultStream.WaitEvent(std::move(smin.RecordEvent()));

        const nutty::cuStream& smax = m_pStreamPool->PeekNextStream();
        nutty::SetStream(smax);
        nutty::Reduce(m_aabbMax.Begin(), dataBuffer, dataBuffer + m_elements, float3max());
        m_defaultStream.WaitEvent(std::move(smax.RecordEvent()));

        float3 os;
        os.z = os.y = os.x = 1;
        float3 mini = *m_aabbMin.Begin();
        float3 maxi = *m_aabbMax.Begin();
        mini = mini - os;
        maxi = maxi + os;

        AABB aabb;
        aabb.min = mini;
        aabb.max = maxi;
        m_nodeAABBs.Insert(0, aabb);

        uint block = 2 * m_elements < 256 ? 2 * m_elements : 256;
        uint grid = nutty::cuda::GetCudaGrid(2 * m_elements, 256U);

        DEVICE_SYNC_CHECK();

        for(int i = 0; i < m_depth-1; ++i)
        {                       
            uint contentSum = 0;
	
            uint copyStartAdd = (1 << i) - 1;
            uint copyLength = ((1 << (i+1)) - 1) - copyStartAdd;

            nutty::Copy(bboxTmp.Begin() + copyStartAdd, m_nodeAABBs.Begin() + copyStartAdd, copyLength);
            nutty::Copy(countTmp.Begin() + copyStartAdd, m_nodeContentCount.Begin() + copyStartAdd, copyLength);

            //m_timer[i].Tick();
 
            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                AABB aabb = bboxTmp[(size_t)(j-1)];
                int axis = getLongestAxis(aabb.min, aabb.max);

                auto begin = m_edgeIndexedEdge.Begin() + contentSum;
                
                uint c = countTmp[(size_t)(j-1)];

                if(c > 1)
                {                
                    const nutty::cuStream& s = m_pStreamPool->PeekNextStream();
                    nutty::SetStream(s);
                    nutty::Sort(begin, begin + c, EdgeSort(axis));
                    
                    DEVICE_SYNC_CHECK();

                    m_defaultStream.WaitEvent(std::move(s.RecordEvent()));
                }
                contentSum += c;
            }

            //m_timer[i].Tock();
   
            //DEBUG_OUT_A("%d %f\n", (i+1), m_timer[i].GetAverageMillis());
            DEVICE_SYNC_CHECK();
 
            reOrderFromEdges<<<grid, block, 0, m_defaultStream()>>>
                (
                m_edgeStructOfArrayArg[(m_toggleBit+1) % 2], m_edgeStructOfArrayArg[m_toggleBit], 2 * m_elements
                );

            DEVICE_SYNC_CHECK();
            
            computeSplits<<<grid, block, 0, m_defaultStream()>>>
                (
                m_nodeAABBs.Begin()(), 
                m_splitStructOfArrayArg[m_toggleBit],
                m_nodeStructOfArrayArg,
                m_edgeStructOfArrayArg[(m_toggleBit+1) % 2],
                m_elements, 
                i, 
                m_perThreadNodePos.Begin()(), 
                m_prefixSumNodeCount.Begin()()
                );

            DEVICE_SYNC_CHECK();
            
            contentSum = 0;

            for(int j = (1 << i); j < (1 << (i+1)); ++j)
            {
                int c = (int)countTmp[(size_t)(j-1)];
                if((int)(c) > 0)
                {
                    auto begin = m_splitSAH.Begin() + contentSum;
                    if(c > 1)
                    {
                        nutty::cuStream& s = m_pStreamPool->PeekNextStream();
                        s.WaitEvent(m_defaultStream.RecordEvent());
                        nutty::SetStream(s);
                        nutty::Reduce(begin, begin + c, ReduceIndexedSAH());
                        m_defaultStream.WaitEvent(std::move(s.RecordEvent()));
                        //printSplit(m_splitStructOfArrayArg, contentSum);
                        DEVICE_SYNC_CHECK();
                    }
                }
                {
                    contentSum += c;
                }
            }

            reOrderFromSAH<<<grid, block, 0, m_defaultStream()>>>
              (
              m_splitStructOfArrayArg[(m_toggleBit+1) % 2], m_splitStructOfArrayArg[m_toggleBit], 2 * m_elements
              );
            
            DEVICE_SYNC_CHECK();

            uint g = 1;
            uint b = (1 << i);

            if(b > 256)
            {
                g = nutty::cuda::GetCudaGrid(b, 256U);
                b = 256;
            }

            nutty::SetStream(m_defaultStream);

            setNodesCount<<<g, b, 0, m_defaultStream()>>>
                (
                m_prefixSumNodeCount.Begin()(), 
                m_splitStructOfArrayArg[(m_toggleBit+1) % 2],
                m_nodeStructOfArrayArg,
                m_elements, 
                i
                );
            
            DEVICE_SYNC_CHECK();

            spreadContent<<<grid, block, 0, m_defaultStream()>>>
                (
                m_nodeStructOfArrayArg,
                m_elements,
                i,
                m_perThreadNodePos.Begin()(),
                m_prefixSumNodeCount.Begin()()
                );
           
            DEVICE_SYNC_CHECK();

            splitNodes<<<g, b, 0, m_defaultStream()>>>
                (
                m_nodeAABBs.Begin()(),
                m_nodeStructOfArrayArg,
                i
                );

            DEVICE_SYNC_CHECK();

            m_toggleBit = (m_toggleBit + 1) % 2;
        }

        uint g = 1;
        uint b = (1 << (m_depth-1));

        if(b > 256)
        {
            g = nutty::cuda::GetCudaGrid(b, 256U);
            b = 256;
        } 

        initLeafs<<<g, b, 0, m_defaultStream()>>>
            (
            m_splitStructOfArrayArg[m_toggleBit],
            m_nodeStructOfArrayArg,
            m_elements,
            m_depth-1
            );
        
        DEVICE_SYNC_CHECK();

        g = nutty::cuda::GetCudaGrid(2 * m_elements, 256U);
        b = 256;

        postProcess<<<g, b, 0, m_defaultStream()>>>
            (
            m_transformedData.Begin()(), 
            m_data->Begin()(), 
            m_edgeStructOfArrayArg[m_toggleBit],
            2 * m_elements
            );

        DEVICE_SYNC_CHECK();
        m_defaultStream.ClearEvents();
        m_pStreamPool->ClearEvents();

        //printBuffer(m_transformedData);
        printBuffer(m_nodeContentCount);
    }

    void Generate2(void)
    {

    }

    void Update(void)
    {
        ClearBuffer();
        Generate();
    }

private:
    void InitBuffer(void)
    {
        m_primBBox.Resize(m_elements);
        m_transformedData.Resize(2 * m_elements);

        hostContentCount.Resize(m_nodesCount);

        m_aabbMin.Resize(m_elements);
        m_aabbMax.Resize(m_elements);

        countTmp.Resize(m_nodesCount);
        bboxTmp.Resize(m_nodesCount);

        m_prefixSumNodeCount.Resize(2 * m_elements);
        m_perThreadNodePos.Resize(2 * m_elements);
        m_nodeAABBs.Resize(m_nodesCount);

        m_null.Resize(max(m_nodesCount, 2 * m_elements));

        nutty::HostBuffer<uint> null(m_null.Size(), 0);

        nutty::Copy(m_null.Begin(), null.Begin(), null.Size());
        nutty::Copy(m_prefixSumNodeCount.Begin(), m_null.Begin(), m_prefixSumNodeCount.Size());
        nutty::Copy(m_perThreadNodePos.Begin(), m_null.Begin(), m_perThreadNodePos.Size());

        uint splitCount = 2 * m_elements;
        m_splitSAH.Resize(splitCount);
        m_splitAxis[0].Resize(splitCount);
        m_splitAxis[1].Resize(splitCount);
        m_splitPrimId[0].Resize(splitCount);
        m_splitPrimId[1].Resize(splitCount);
        m_splitSplitT[0].Resize(splitCount);
        m_splitSplitT[1].Resize(splitCount);
        m_splitBelow[0].Resize(splitCount);
        m_splitBelow[1].Resize(splitCount);
        m_splitAbove[0].Resize(splitCount);
        m_splitAbove[1].Resize(splitCount);

        m_splitStructOfArrayArg[0].sah = m_splitSAH.GetDevicePtr()();
        m_splitStructOfArrayArg[0].above = m_splitAbove[0].GetDevicePtr()();
        m_splitStructOfArrayArg[0].below = m_splitBelow[0].GetDevicePtr()();
        m_splitStructOfArrayArg[0].split = m_splitSplitT[0].GetDevicePtr()();
        m_splitStructOfArrayArg[0].primId = m_splitPrimId[0].GetDevicePtr()();
        m_splitStructOfArrayArg[0].axis = m_splitAxis[0].GetDevicePtr()();

        m_splitStructOfArrayArg[1].sah = m_splitSAH.GetDevicePtr()();
        m_splitStructOfArrayArg[1].above = m_splitAbove[1].GetDevicePtr()();
        m_splitStructOfArrayArg[1].below = m_splitBelow[1].GetDevicePtr()();
        m_splitStructOfArrayArg[1].split = m_splitSplitT[1].GetDevicePtr()();
        m_splitStructOfArrayArg[1].primId = m_splitPrimId[1].GetDevicePtr()();
        m_splitStructOfArrayArg[1].axis = m_splitAxis[1].GetDevicePtr()();

        uint edgeCount = m_elements * 2;
        m_edgeType[0].Resize(edgeCount);
        m_edgeType[1].Resize(edgeCount);
        m_edgePrimId[0].Resize(edgeCount);
        m_edgePrimId[1].Resize(edgeCount);
        m_edgeIndexedEdge.Resize(edgeCount);

        m_edgeStructOfArrayArg[0].type = m_edgeType[0].GetDevicePtr()();
        m_edgeStructOfArrayArg[0].primId = m_edgePrimId[0].GetDevicePtr()();
        m_edgeStructOfArrayArg[0].indexedEdge = m_edgeIndexedEdge.GetDevicePtr()();
        m_edgeStructOfArrayArg[1].type = m_edgeType[1].GetDevicePtr()();
        m_edgeStructOfArrayArg[1].primId = m_edgePrimId[1].GetDevicePtr()();
        m_edgeStructOfArrayArg[1].indexedEdge = m_edgeIndexedEdge.GetDevicePtr()();

        m_nodeSplitAxis.Resize(m_nodesCount);
        m_nodeIsLeaf.Resize(m_nodesCount);
        m_nodeContentStartIndex.Resize(m_nodesCount);
        m_nodeSplit.Resize(m_nodesCount);
        m_nodeContentCount.Resize(m_nodesCount);
        m_nodeBelow.Resize(m_nodesCount);
        m_nodeContentCount.Insert(0, m_elements * 2);

        m_nodesBuffer.Resize(1);

        m_nodeStructOfArrayArg.axis = m_nodeSplitAxis.GetDevicePtr()();
        m_nodeStructOfArrayArg.contentStartIndex = m_nodeContentStartIndex.GetDevicePtr()();
        m_nodeStructOfArrayArg.contentCount = m_nodeContentCount.GetDevicePtr()();
        m_nodeStructOfArrayArg.leaf = m_nodeIsLeaf.GetDevicePtr()();
        m_nodeStructOfArrayArg.split = m_nodeSplit.GetDevicePtr()();
        m_nodeStructOfArrayArg.below = m_nodeBelow.GetDevicePtr()();

        m_nodesBuffer.Insert(0, m_nodeStructOfArrayArg);

        ClearBuffer();
    }

public:
    void ClearBuffer(void)
    {
        nutty::ZeroMem(m_prefixSumNodeCount);
        nutty::ZeroMem(m_perThreadNodePos);
        nutty::Copy(m_nodeContentCount.Begin()+1, m_null.Begin(), m_nodeContentCount.Size()-1);

        nutty::ZeroMem(m_nodeAABBs);
        nutty::ZeroMem(m_nodeIsLeaf);
        nutty::ZeroMem(m_nodeContentStartIndex);
        nutty::ZeroMem(m_nodeSplit);
        nutty::ZeroMem(m_nodeSplitAxis);
        nutty::ZeroMem(m_nodeBelow);

        nutty::ZeroMem(m_splitAbove[0]);
        nutty::ZeroMem(m_splitBelow[0]);
        nutty::ZeroMem(m_splitAxis[0]);
        nutty::ZeroMem(m_splitPrimId[0]);
        nutty::ZeroMem(m_splitSAH);
        nutty::ZeroMem(m_splitSplitT[0]);
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
