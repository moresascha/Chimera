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
#include "DoubleBuffer.h"

#include "cuTimer.cuh"
#include "MemoryPool.h"

#include "kd_kernel.cuh"

#include <sstream>

#include <set>

bool g_treeDebug = false; 

std::stringstream g_stringSream;

void _print(const char* str)
{
    OutputDebugStringA(str);
}

std::ostream& operator<<(std::ostream &out, const IndexedSplit& t)
{
    out << "IndexedSplit [" << t.sah << ", " << t.index << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const IndexedEdge& t)
{
    out << "IndexedEdge [" << t.v << ", " << t.index << "]";
    return out;
}

std::ostream& operator<<(std::ostream &out, const BBox& t)
{
    out << "BBox [" << t.min.x << "," << t.min.y << "," << t.min.z << "|" << t.max.x << "," << t.max.y << "," << t.max.z << "]";
    return out;
}

template <
    typename T
>
void print(T val)
{
    g_stringSream.str("");
    g_stringSream << val << ", ";
    _print(g_stringSream.str().c_str());
}

void printC(byte val)
{
    g_stringSream.str("");
    g_stringSream << (int)val << ", ";
    _print(g_stringSream.str().c_str());
}

template<>
struct ShrdMemory<IndexedSplit>
{
    __device__ IndexedSplit* Ptr(void) 
    { 
        extern __device__ __shared__ IndexedSplit s_split[];
        return s_split;
    }
};

template<>
struct ShrdMemory<IndexedEdge>
{
    __device__ IndexedEdge* Ptr(void) 
    { 
        extern __device__ __shared__ IndexedEdge s_edge[];
        return s_edge;
    }
};

struct ReduceIndexedSplit
{
    __device__ __host__ IndexedSplit operator()(IndexedSplit t0, IndexedSplit t1)
    {
        return t0.sah < t1.sah ? t0 : t1;
    }
};

struct SplitSort
{
    __device__ IndexedSplit operator()(IndexedSplit t0, IndexedSplit t1)
    {
        return t0.sah < t1.sah ? t0 : t1;
    }
};

struct EdgeSort
{
    __device__ char operator()(IndexedEdge t0, IndexedEdge t1)
    {
        return t0.v > t1.v;
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

void printBufferChar(nutty::DeviceBuffer<byte>& buffer)
{
    if(g_treeDebug)
    {
        nutty::ForEach(buffer.Begin(), buffer.End(), printC);
       _print("\n");
    }
}

template <
    typename T
>
void printBuffer(nutty::DeviceBuffer<T>& buffer)
{
    if(g_treeDebug)
    {
        nutty::ForEach(buffer.Begin(), buffer.End(), print<T>);
        _print("\n");
    }
}

#ifdef _DEBUG
#define PRINT_BUFFER(__buffer) _print(#__buffer); _print(":\n"); printBuffer(__buffer)
#else
#define PRINT_BUFFER(__buffer)
#endif

template <
    typename T
>
class kdTree : public IKDTree
{

public:
    uint m_nodesCount;
    uint m_elementCount;
    byte m_depth;
    byte m_maxDepth;

    nutty::cuModule* m_cudaModule;

    nutty::DeviceBuffer<T>* m_primitives;

    nutty::DeviceBuffer<uint> m_edgeMask;
    nutty::DeviceBuffer<uint> m_scannedEdgeMask;
    nutty::DeviceBuffer<uint> m_edgeMaskSums;
    
    nutty::DeviceBuffer<float3> m_bboxMin;
    nutty::DeviceBuffer<float3> m_bboxMax;

    nutty::HostBuffer<uint> m_hNodesContentCount;

    Node m_nodes;
    nutty::DeviceBuffer<BBox> m_nodesBBox;
    nutty::DeviceBuffer<byte> m_nodesIsLeaf;
    nutty::DeviceBuffer<byte> m_nodesSplitAxis;
    nutty::DeviceBuffer<float> m_nodesSplit;
    nutty::DeviceBuffer<uint> m_nodesContentCount;
    nutty::DeviceBuffer<uint> m_nodesStartAdd;

    Split m_splits[2];
    nutty::DeviceBuffer<IndexedSplit> m_splitsIndexedSplit;
    DoubleBuffer<float> m_splitsSplit;
    DoubleBuffer<byte> m_splitsAxis;
    DoubleBuffer<uint> m_splitsAbove;
    DoubleBuffer<uint> m_splitsBelow;

    Edge m_edges[2];
    nutty::DeviceBuffer<IndexedEdge> m_edgesIndexedEdge;
    DoubleBuffer<byte> m_edgesType;
    DoubleBuffer<uint> m_edgesNodeIndex;
    DoubleBuffer<uint> m_edgesPrimId;
    DoubleBuffer<uint> m_edgesPrefixSum;

    Primitive m_nodesContent;
    nutty::DeviceBuffer<uint> m_primIndex;
    nutty::DeviceBuffer<uint> m_primNodeIndex;
    nutty::DeviceBuffer<uint> m_primPrefixSum;

    nutty::DeviceBuffer<BBox> m_primAABBs;

    nutty::cuStreamPool* m_pStreamPool;

    nutty::cuStream m_defaultStream;

    cuTimer m_timer[16];

    std::stringstream _stream;

    kdTree(byte depth, byte maxDepth);

    void GetContentCountStr(std::string& s);

    uint GetCurrentDepth(void);

    void* GetData(void);

    Node GetNodes(void);

    nutty::cuStream& GetDefaultStream(void);

    nutty::DeviceBuffer<BBox>* GetAABBs(void);

    void SetDepth(uint d);

    void Init(void* data, uint elements);

    void Generate(void);

    void Update(void);

    void ClearBuffer(void);

    void InitBuffer(void);

    void GrowMemory(void);
    
    ~kdTree(void);
};

template <typename T>
void kdTree<T>::InitBuffer(void)
{
    m_nodesBBox.Resize(m_nodesCount);
    m_nodesIsLeaf.Resize(m_nodesCount);
    m_nodesSplit.Resize(m_nodesCount);
    m_nodesStartAdd.Resize(m_nodesCount);
    m_nodesSplitAxis.Resize(m_nodesCount);
    m_nodesContentCount.Resize(m_nodesCount);
    m_hNodesContentCount.Resize(elemsOnLevel(m_maxDepth-1));

    m_nodes.aabb = m_nodesBBox.GetDevicePtr()();
    m_nodes.isLeaf = m_nodesIsLeaf.GetDevicePtr()();
    m_nodes.splitAxis = m_nodesSplitAxis.GetDevicePtr()();
    m_nodes.split = m_nodesSplit.GetDevicePtr()();
    m_nodes.contentStart = m_nodesStartAdd.GetDevicePtr()();
    m_nodes.contentCount = m_nodesContentCount.GetDevicePtr()();

    m_bboxMin.Resize(m_elementCount); nutty::ZeroMem(m_bboxMin);
    m_bboxMax.Resize(m_elementCount); nutty::ZeroMem(m_bboxMax);
    m_primAABBs.Resize(m_elementCount); nutty::ZeroMem(m_primAABBs);

    GrowMemory();

    ClearBuffer();
}

template <typename T>
void kdTree<T>::ClearBuffer(void)
{
    nutty::ZeroMem(m_edgeMask);
    nutty::ZeroMem(m_scannedEdgeMask);
    nutty::ZeroMem(m_edgeMaskSums);

    nutty::ZeroMem(m_nodesBBox);
    nutty::ZeroMem(m_nodesContentCount);
    nutty::ZeroMem(m_nodesIsLeaf);
    nutty::ZeroMem(m_nodesSplit);
    nutty::ZeroMem(m_nodesStartAdd);
    nutty::ZeroMem(m_nodesSplitAxis);

    m_splitsAbove.ZeroMem();
    m_splitsBelow.ZeroMem();
    m_splitsAxis.ZeroMem();
    m_splitsSplit.ZeroMem();

    m_edgesNodeIndex.ZeroMem();
    m_edgesPrimId.ZeroMem();
    m_edgesType.ZeroMem();
    m_edgesPrefixSum.ZeroMem();
}

template <typename T>
void kdTree<T>::GrowMemory(void)
{
    _print("Growing Memory!\n");
    uint edgeCount = 4 * m_elementCount; //4 times as big
    m_primIndex.Resize(edgeCount);
    m_primNodeIndex.Resize(edgeCount);
    m_primPrefixSum.Resize(edgeCount);

    m_nodes.content = m_primIndex.GetDevicePtr()();

    m_nodesContent.primIndex = m_primIndex.GetDevicePtr()();
    m_nodesContent.nodeIndex = m_primNodeIndex.GetDevicePtr()();
    m_nodesContent.prefixSum = m_primPrefixSum.GetDevicePtr()();

    m_splitsAbove.Resize(edgeCount);
    m_splitsBelow.Resize(edgeCount);
    m_splitsAxis.Resize(edgeCount);
    m_splitsSplit.Resize(edgeCount);
    m_splitsIndexedSplit.Resize(edgeCount);
    
    m_splits[0].above = m_splitsAbove.Get(0).GetDevicePtr()();
    m_splits[0].below = m_splitsBelow.Get(0).GetDevicePtr()();
    m_splits[0].axis = m_splitsAxis.Get(0).GetDevicePtr()();
    m_splits[0].indexedSplit = m_splitsIndexedSplit.GetDevicePtr()();
    m_splits[0].v = m_splitsSplit.Get(0).GetDevicePtr()();

    m_splits[1].above = m_splitsAbove.Get(1).GetDevicePtr()();
    m_splits[1].below = m_splitsBelow.Get(1).GetDevicePtr()();
    m_splits[1].axis = m_splitsAxis.Get(1).GetDevicePtr()();
    m_splits[1].indexedSplit = m_splitsIndexedSplit.GetDevicePtr()();
    m_splits[1].v = m_splitsSplit.Get(1).GetDevicePtr()();

    m_edgesIndexedEdge.Resize(edgeCount);
    m_edgesNodeIndex.Resize(edgeCount);
    m_edgesPrimId.Resize(edgeCount);
    m_edgesType.Resize(edgeCount);
    m_edgesPrefixSum.Resize(edgeCount);

    m_edges[0].indexedEdge = m_edgesIndexedEdge.GetDevicePtr()();
    m_edges[0].nodeIndex = m_edgesNodeIndex.Get(0).GetDevicePtr()();
    m_edges[0].primId = m_edgesPrimId.Get(0).GetDevicePtr()();
    m_edges[0].type = m_edgesType.Get(0).GetDevicePtr()();
    m_edges[0].prefixSum = m_edgesPrefixSum.Get(0).GetDevicePtr()();

    m_edges[1].indexedEdge = m_edgesIndexedEdge.GetDevicePtr()();
    m_edges[1].nodeIndex = m_edgesNodeIndex.Get(1).GetDevicePtr()();
    m_edges[1].primId = m_edgesPrimId.Get(1).GetDevicePtr()();
    m_edges[1].type = m_edgesType.Get(1).GetDevicePtr()();
    m_edges[1].prefixSum = m_edgesPrefixSum.Get(1).GetDevicePtr()();

    m_edgeMask.Resize(edgeCount);
    m_scannedEdgeMask.Resize(edgeCount);
    m_edgeMaskSums.Resize(edgeCount); //way to big but /care
}

template <typename T>
void kdTree<T>::Generate(void)
{
    static float3 max3f = {FLT_MAX, FLT_MAX, FLT_MAX};
    static float3 min3f = -max3f;
    
    //byte toggleBit = 0;

    m_elementCount = (uint)m_primitives->Size();

    m_nodesContentCount.Insert(0, m_elementCount);

    uint _block = m_elementCount < 256 ? m_elementCount : 256;
    uint _grid = nutty::cuda::GetCudaGrid(m_elementCount, _block);

    computePerPrimBBox<<<_grid, _block>>>
            (
            m_primitives->Begin()(), m_primAABBs.Begin()(), SphereBBox(), m_elementCount
            );

    DEVICE_SYNC_CHECK();

    initNodesContent<<<_grid, _block>>>
        (
        m_nodesContent, m_elementCount
        );

    DEVICE_SYNC_CHECK();

    nutty::Reduce(m_bboxMin.Begin(), m_primitives->Begin(), m_primitives->End(), float3min(), max3f);
    nutty::Reduce(m_bboxMax.Begin(), m_primitives->Begin(), m_primitives->End(), float3max(), min3f);

    DEVICE_SYNC_CHECK();

    BBox bbox;
    bbox.min = m_bboxMin[0];
    bbox.max = m_bboxMax[0];

    bbox.min.x -= 1;
    bbox.min.y -= 1;
    bbox.min.z -= 1;

    bbox.max.x += 1;
    bbox.max.y += 1;
    bbox.max.z += 1;

    m_nodesBBox.Insert(0, bbox);

    for(byte d = 0; d < m_depth; ++d)
    {
        DEBUG_OUT_A("---------------------------- Level: %d ----------------------------\n", (int)d+1);
        uint elementBlock = m_elementCount < 256 ? m_elementCount : 256;
        uint elementGrid = nutty::cuda::GetCudaGrid(m_elementCount, elementBlock);

        nutty::ZeroMem(m_edgeMask);
        nutty::ZeroMem(m_scannedEdgeMask);

        Edge edgesSrc = m_edges[0];
        Edge edgesDst = m_edges[1];
        Split splitsSrc = m_splits[0];
        Split splitsDst = m_splits[1];
        
        DEVICE_SYNC_CHECK();

        createEdges<<<elementGrid, elementBlock>>>
            (
            edgesSrc, m_nodes, m_primAABBs.Begin()(), m_nodesContent, d, m_elementCount
            );

        DEBUG_OUT_A("%d edges created!", 2 * m_elementCount);
        DEVICE_SYNC_CHECK();
        uint copyStart = (1 << d) - 1;
        uint copyLength = 1 << d;

        nutty::Copy(m_hNodesContentCount.Begin(), m_nodesContentCount.Begin() + copyStart, copyLength);

        PRINT_BUFFER(m_nodesContentCount);

        uint start = 0;
        for(int i = 0; i < copyLength; ++i)
        {
            DEBUG_OUT_A("%d ", m_hNodesContentCount[i]);
        }
        DEBUG_OUT("\n");

        for(int i = 0; i < (1 << d); ++i)
        {
            uint length = 2 * m_hNodesContentCount[i];
            if(length == 0)
            {
                continue;
            }
            DEBUG_OUT_A("%d %d %d\n", m_elementCount * 2, start, start + length);

            assert(m_elementCount * 2 >= start + length);
            
            nutty::Sort(m_edgesIndexedEdge.Begin() + start, m_edgesIndexedEdge.Begin() + start + length, EdgeSort());

            DEVICE_SYNC_CHECK();

            start += length;
        }

        uint edgeCount = 2 * m_elementCount;
        uint edgeBlock = edgeCount < 256 ? edgeCount : 256;
        uint edgeGrid = nutty::cuda::GetCudaGrid(edgeCount, edgeBlock);

        reorderEdges<<<edgeGrid, edgeBlock>>>
            (
            edgesDst, edgesSrc, edgeCount
            );

        DEVICE_SYNC_CHECK();

        computeSAHSplits<<<edgeGrid, edgeBlock>>>
            (
            edgesDst, m_nodes, splitsSrc, m_nodesContent, d, edgeCount
            );

        DEVICE_SYNC_CHECK();

        start = 0;

        for(int i = 0; i < (1 << d); ++i)
        {
            uint length = 2 * m_hNodesContentCount[i];
            if(length == 0)
            {
                continue;
            }

            IndexedSplit neutralSplit;
            neutralSplit.index = 0;
            neutralSplit.sah = FLT_MAX;

            nutty::Reduce(m_splitsIndexedSplit.Begin() + start, m_splitsIndexedSplit.Begin() + start + length, ReduceIndexedSplit(), neutralSplit);

            DEVICE_SYNC_CHECK();

            start += length;
        }
        
        reorderSplits<<<edgeGrid, edgeBlock>>> //dont need to reorder all
            (
            splitsDst, splitsSrc, edgeCount
            );
        
        start = 0;

        for(int i = 0; i < (1 << d); ++i)
        {
            uint length = 2 * m_hNodesContentCount[i];
            if(length == 0)
            {
                continue;
            }
            float s = *(m_splitsSplit.Get(1).Begin() + start);
            uint above = *(m_splitsAbove.Get(1).Begin() + start);
            uint below = *(m_splitsBelow.Get(1).Begin() + start);
            m_nodesSplit.Insert((1 << d) + i - 1, s);
            DEBUG_OUT_A("split=%f, %d, %d\n", s, below, above);
            start += length;
        }

        DEVICE_SYNC_CHECK();

        processEdges<<<edgeGrid, edgeBlock>>>
            (
            m_nodes, edgesDst, m_edgeMask.Begin()(), d, edgeCount
            );

        DEVICE_SYNC_CHECK();

        nutty::ExclusivePrefixSumScan(m_edgeMask.Begin(), m_edgeMask.Begin() + edgeCount, m_scannedEdgeMask.Begin(), m_edgeMaskSums.Begin());

        DEVICE_SYNC_CHECK();

        uint nodeCount = 1 << d;
        uint nodeBlock = nodeCount < 256 ? nodeCount : 256;
        uint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);
        createNodes<<<nodeGrid, nodeBlock>>>
            (
            m_scannedEdgeMask.Begin()(), m_edgeMask.Begin()(), splitsDst, m_nodes, d, d == (m_depth - 1)
            );

        DEVICE_SYNC_CHECK();

        compactContentFromEdges<<<edgeGrid, edgeBlock>>>
            (
            edgesDst, m_nodes, m_nodesContent, m_edgeMask.Begin()(), m_scannedEdgeMask.Begin()(), d, edgeCount
            );

        DEVICE_SYNC_CHECK();

        DEBUG_OUT_A("m_elementCount=%d -> ", m_elementCount);

        m_elementCount = m_scannedEdgeMask[edgeCount - 1] + m_edgeMask[edgeCount - 1];

        DEBUG_OUT_A("m_elementCount=%d\n", m_elementCount);

        if(2 * m_elementCount > m_edgeMask.Size() && d < m_depth-1)
        {
            GrowMemory();
        }

        DEVICE_SYNC_CHECK();
    }
}

template <typename T>
kdTree<T>::kdTree(byte depth, byte maxDepth) : 
    m_cudaModule(NULL), 
    m_depth(depth), 
    m_maxDepth(maxDepth), 
    m_nodesCount(0), 
    m_elementCount(0)
{    
    assert(maxDepth >= depth);
    m_pStreamPool = new nutty::cuStreamPool();
}

template <typename T>
void kdTree<T>::GetContentCountStr(std::string& s)
{
//     uint limit = 32;
//     nutty::Copy(hostContentCount.Begin(), m_contentContentCount.Begin(), min(m_nodesCount, limit));
//     s.clear();
//     _stream.str("");
//     _stream << "Prims:" << m_elements << " - ";
//     for(uint i = 0; i < min(m_nodesCount, limit); ++i)
//     {
//         _stream << hostContentCount[i] << " ";
//     }
// 
//     s = _stream.str();
}

template <typename T>
uint kdTree<T>::GetCurrentDepth(void)
{
    return m_depth;
}

template <typename T>
void* kdTree<T>::GetData(void)
{
    return m_primitives;
}

template <typename T>
Node kdTree<T>::GetNodes(void)
{
    return m_nodes;
}

template <typename T>
nutty::cuStream& kdTree<T>::GetDefaultStream(void)
{
    return m_defaultStream;
}

template <typename T>
nutty::DeviceBuffer<BBox>* kdTree<T>::GetAABBs(void)
{
    return &m_nodesBBox;
}

template <typename T>
void kdTree<T>::SetDepth(uint d)
{
    m_depth = min(m_maxDepth, max(2, d));
}

template <typename T>
kdTree<T>::~kdTree(void)
{
    SAFE_DELETE(m_primitives);
    SAFE_DELETE(m_pStreamPool);
}

template <typename T>
void kdTree<T>::Init(void* prims, uint elements)
{
    m_elementCount = elements;

    m_nodesCount = (1 << (uint)m_maxDepth) - 1;

    m_primitives = (nutty::DeviceBuffer<T>*)(prims);
        
    InitBuffer();
}

template <typename T>
void kdTree<T>::Update(void)
{
    ClearBuffer();
    Generate();
}

kdTree<float3>* g_tree;

extern "C" void init(void)
{
    nutty::Init();
}

extern "C" void release(void)
{
    if(g_tree)
    {
        delete g_tree;
    }
    nutty::Release();
}

extern "C" IKDTree* generate(nutty::DeviceBuffer<float3>* data, uint elements, uint d, uint maxDepth)
{
    g_tree = new kdTree<float3>(d, maxDepth);
    g_tree->Init((void*)data, elements);
    g_tree->Generate();
    return g_tree;
}
