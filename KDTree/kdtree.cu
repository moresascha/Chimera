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

#include "kd_kernel.h"

#include <sstream>

#include <set>

#include "veb.h"

bool g_treeDebug = false; 

std::stringstream g_stringSream;

#undef PROFILE

#ifdef PROFILE
uint g_profilingSteps = 0;
#endif

uint putpos = 0;

static void dfs(int node, int depth, nutty::HostBuffer<uint>& d)
{   
    if(depth == 1)
    {
        return;
    }
    d[putpos++] = 1 + node;
    d[putpos++] = (node + (1 << (depth-1)));
        
    dfs(node + 1, depth - 1, d);
        
    dfs(node + (1 << (depth-1)), depth - 1, d);
}

void _print(const char* str)
{
    if(!g_treeDebug)
    {
        return;
    }
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
    typename T,
    typename char trim
>
void __print(T val)
{
    g_stringSream.str("");
    g_stringSream << val << trim;
    _print(g_stringSream.str().c_str());
}

#define print __print

void __printC(byte val)
{
    g_stringSream.str("");
    g_stringSream << (int)val << " ";
    _print(g_stringSream.str().c_str());
}

#define printC __printC

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
struct ShrdMemory<BBox>
{
    __device__ BBox* Ptr(void) 
    { 
        extern __device__ __shared__ BBox s_bbox[];
        return s_bbox;
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

struct BBoxReducer
{
    __device__  BBox operator()(BBox& t0, BBox& t1)
    {
        BBox bbox;
        bbox.min = fminf(t0.min, t1.min);
        bbox.max = fmaxf(t0.max, t1.max);
        return bbox;
    }
};

struct SphereBBox
{
    float bhe;
    SphereBBox(float _bhe)
    {
        bhe = _bhe;
    }

    __device__ BBox operator()(float3 pos)
    {
        BBox bbox;
        bbox.min = pos - make_float3(bhe, bhe, bhe);
        bbox.max = pos + make_float3(bhe, bhe, bhe);
        return bbox;
    }
};

struct TriangleBBox
{
    __device__ BBox operator()(float3 pos)
    {
        BBox bbox;
        return bbox;
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
    typename char trim,
    typename T
>
void printBuffer(nutty::DeviceBuffer<T>& buffer)
{
    if(g_treeDebug)
    {
        for(int i = 0; i < buffer.Size(); ++i)
        {
            T t = buffer[i];
            DEBUG_OUT_A("[%d] ", i);
            print<T, trim>(t);
        }
        //nutty::ForEach(buffer.Begin(), buffer.End(), print<T, trim>);
        _print("\n");
    }
}

#ifdef _DEBUG
#define PRINT_BUFFER(__buffer) _print(#__buffer); _print(":\n"); printBuffer<', '>(__buffer)
#define PRINT_BUFFER_N(__buffer) _print(#__buffer); _print(":\n"); printBuffer<'\n'>(__buffer)
#else
#define PRINT_BUFFER(__buffer)
#define PRINT_BUFFER_N(__buffer)
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
    float m_sphereRadius;

    nutty::cuModule* m_cudaModule;

    nutty::DeviceBuffer<uint> m_depthFirstMask;

    nutty::DeviceBuffer<T>* m_primitives;
    nutty::DeviceBuffer<T> m_tprimitives;

    nutty::DeviceBuffer<uint> m_edgeMask;
    nutty::DeviceBuffer<uint> m_scannedEdgeMask;
    nutty::DeviceBuffer<uint> m_edgeMaskSums;
    
    nutty::DeviceBuffer<float3> m_bboxMin;
    nutty::DeviceBuffer<float3> m_bboxMax;
    nutty::DeviceBuffer<BBox> m_sceneBBox;

    nutty::HostBuffer<uint> m_hNodesContentCount;

    Node m_nodes;
    nutty::DeviceBuffer<BBox> m_nodesBBox;
    nutty::DeviceBuffer<byte> m_nodesIsLeaf;
    nutty::DeviceBuffer<byte> m_nodesSplitAxis;
    nutty::DeviceBuffer<float> m_nodesSplit;
    nutty::DeviceBuffer<uint> m_nodesContentCount;
    nutty::DeviceBuffer<uint> m_nodesStartAdd;
    nutty::DeviceBuffer<uint> m_nodesAbove;
    nutty::DeviceBuffer<uint> m_nodesBelow;

    Node m_dfoNodes;
    nutty::DeviceBuffer<byte> m_dfoNodesIsLeaf;
    nutty::DeviceBuffer<byte> m_dfoNodesSplitAxis;
    nutty::DeviceBuffer<float> m_dfoNodesSplit;
    nutty::DeviceBuffer<uint> m_dfoNodesContentCount;
    nutty::DeviceBuffer<uint> m_dfoNodesStartAdd;

    Split m_splits;
    nutty::DeviceBuffer<IndexedSplit> m_splitsIndexedSplit;
    nutty::DeviceBuffer<float> m_splitsSplit;
    nutty::DeviceBuffer<byte> m_splitsAxis;
    nutty::DeviceBuffer<uint> m_splitsAbove;
    nutty::DeviceBuffer<uint> m_splitsBelow;

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
    nutty::DeviceBuffer<BBox> m_tPrimAABBs;

    nutty::cuStreamPool* m_pStreamPool;

    nutty::cuStream m_defaultStream;

    cuTimer m_timer[32];

    std::stringstream _stream;

    PrimType m_primType;

    kdTree(byte depth, byte maxDepth);

    void GetContentCountStr(std::string& s);

    uint GetCurrentDepth(void);

    void* GetData(void);

    Node GetNodes(void);

    nutty::cuStream& GetDefaultStream(void);

    nutty::DeviceBuffer<BBox>* GetAABBs(void);

    nutty::DeviceBuffer<BBox>* GetPrimAABBs(void);

    void SetDepth(uint d);

    void Init(void* data, uint elements, PrimType type, float radius);

    uint GetNodesCount(void) { return m_nodesCount; }

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
    m_nodesAbove.Resize(m_nodesCount);
    m_nodesBelow.Resize(m_nodesCount);
    m_hNodesContentCount.Resize(elemsOnLevel(m_maxDepth-1));

//     m_dfoNodesIsLeaf.Resize(m_nodesCount);
//     m_dfoNodesSplitAxis.Resize(m_nodesCount);
//     m_dfoNodesSplit.Resize(m_nodesCount);
//     m_dfoNodesContentCount.Resize(m_nodesCount);
//     m_dfoNodesStartAdd.Resize(m_nodesCount);

//     m_dfoNodes.isLeaf = m_dfoNodesIsLeaf.GetDevicePtr()();
//     m_dfoNodes.splitAxis = m_dfoNodesSplitAxis.GetDevicePtr()();
//     m_dfoNodes.split = m_dfoNodesSplit.GetDevicePtr()();
//     m_dfoNodes.contentCount = m_dfoNodesContentCount.GetDevicePtr()();
//     m_dfoNodes.contentStart = m_dfoNodesStartAdd.GetDevicePtr()();

    m_nodes.aabb = m_nodesBBox.GetDevicePtr()();
    m_nodes.isLeaf = m_nodesIsLeaf.GetDevicePtr()();
    m_nodes.splitAxis = m_nodesSplitAxis.GetDevicePtr()();
    m_nodes.split = m_nodesSplit.GetDevicePtr()();
    m_nodes.contentStart = m_nodesStartAdd.GetDevicePtr()();
    m_nodes.contentCount = m_nodesContentCount.GetDevicePtr()();
    m_nodes.below = m_nodesBelow.GetDevicePtr()();
    m_nodes.above = m_nodesAbove.GetDevicePtr()();

    /*m_bboxMin.Resize(m_elementCount); nutty::ZeroMem(m_bboxMin);
    m_bboxMax.Resize(m_elementCount); nutty::ZeroMem(m_bboxMax);*/
    m_primAABBs.Resize(m_elementCount); nutty::ZeroMem(m_primAABBs);
    m_sceneBBox.Resize(m_elementCount); nutty::ZeroMem(m_sceneBBox);

    GrowMemory();

//     nutty::HostBuffer<uint> b(GetNodesCount());
//     dfs(0, m_maxDepth, b);
// 
//     for(int i = 0; i < b.Size(); ++i)
//     {
//         DEBUG_OUT_A("%d ", b[i]);
//     }
//     DEBUG_OUT("\n");
//     m_depthFirstMask.Resize(GetNodesCount());
// 
//     nutty::Copy(m_depthFirstMask.Begin(), b.Begin(), GetNodesCount());

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
    nutty::ZeroMem(m_nodesAbove);
    nutty::ZeroMem(m_nodesBelow);

    nutty::ZeroMem(m_splitsAbove);
    nutty::ZeroMem(m_splitsBelow);
    nutty::ZeroMem(m_splitsAxis);
    nutty::ZeroMem(m_splitsSplit);

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
    
    m_splits.above = m_splitsAbove.GetDevicePtr()();
    m_splits.below = m_splitsBelow.GetDevicePtr()();
    m_splits.axis = m_splitsAxis.GetDevicePtr()();
    m_splits.indexedSplit = m_splitsIndexedSplit.GetDevicePtr()();
    m_splits.v = m_splitsSplit.GetDevicePtr()();

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
    
#if defined PROFILE
    uint timerIndex = 0;
    cuTimer& bboxTimer = m_timer[timerIndex++];
    bboxTimer.Tick();
#endif
    computePerPrimBBox<<<_grid, _block>>>
            (
            m_primitives->Begin()(), m_primAABBs.Begin()(), SphereBBox(m_sphereRadius), m_elementCount
            );
    
#if defined PROFILE
    bboxTimer.Tock();

#endif
  

    DEVICE_SYNC_CHECK();

#if defined PROFILE
    cuTimer& initNodesTimer = m_timer[timerIndex++];
    initNodesTimer.Tick();
#endif

    initNodesContent<<<_grid, _block>>>
        (
        m_nodesContent, m_elementCount
        );

#if defined PROFILE
    initNodesTimer.Tock();
#endif

    DEVICE_SYNC_CHECK();
 
/*    nutty::Reduce(m_bboxMin.Begin(), m_primitives->Begin(), m_primitives->End(), float3min(), max3f);
    nutty::Reduce(m_bboxMax.Begin(), m_primitives->Begin(), m_primitives->End(), float3max(), min3f);
    */

    BBox bboxN;
    bboxN.min = max3f;
    bboxN.max = min3f;

#if defined PROFILE
    cuTimer& reduceBBox = m_timer[timerIndex++];
    reduceBBox.Tick();
#endif

    nutty::Reduce(m_sceneBBox.Begin(), m_primAABBs.Begin(), m_primAABBs.End(), BBoxReducer(), bboxN);

#if defined PROFILE
    reduceBBox.Tock();
#endif    

    DEVICE_SYNC_CHECK();

    nutty::Copy(m_nodesBBox.Begin(), m_sceneBBox.Begin(), 1);

    m_nodesIsLeaf.Insert(0, m_depth == 0);

    for(byte d = 0; d < m_depth-1; ++d)
    {
        //DEBUG_OUT_A("---------------------------- Level: %d ----------------------------\n", (int)d+1);
        uint elementBlock = m_elementCount < 256 ? m_elementCount : 256;
        uint elementGrid = nutty::cuda::GetCudaGrid(m_elementCount, elementBlock);

        nutty::ZeroMem(m_edgeMask);
        nutty::ZeroMem(m_scannedEdgeMask);

        Edge edgesSrc = m_edges[0];
        Edge edgesDst = m_edges[1];
        
        DEVICE_SYNC_CHECK();

#if defined PROFILE
        timerIndex = 3;
        cuTimer& createEdgesTimer = m_timer[timerIndex++];
        createEdgesTimer.Tick();
#endif
        createEdges<<<elementGrid, elementBlock>>>
            (
            edgesSrc, m_nodes, m_primAABBs.Begin()(), m_nodesContent, d, m_elementCount
            );

#if defined PROFILE
        createEdgesTimer.Tock();
#endif    
        DEVICE_SYNC_CHECK();

#ifndef DYNAMIC_PARALLELISM
        uint copyStart = (1 << d) - 1;
        uint copyLength = 1 << d;

        nutty::Copy(m_hNodesContentCount.Begin(), m_nodesContentCount.Begin() + copyStart, copyLength);
#endif

        uint start = 0;

#if defined PROFILE
        cuTimer& sortTimer = m_timer[timerIndex++];
        sortTimer.Tick();
#endif

        uint nodeCount = 1 << d;
        uint nodeBlock = nodeCount < 256 ? nodeCount : 256;
        uint nodeGrid = nutty::cuda::GetCudaGrid(nodeCount, nodeBlock);

#if defined DYNAMIC_PARALLELISM

        startSort<<<nodeGrid, nodeBlock>>>
            (
            m_nodes, 
            m_edgesIndexedEdge.Begin()(),
            (1 << d) - 1
            );
#else
        for(int i = 0; i < (1 << d); ++i)
        {
            uint length = 2 * m_hNodesContentCount[i];
            if(length == 0)
            {
                continue;
            }
            //DEBUG_OUT_A("%d %d %d\n", m_elementCount * 2, start, start + length);
            //nutty::SetStream(m_pStreamPool->PeekNextStream());

            nutty::Sort(m_edgesIndexedEdge.Begin() + start, m_edgesIndexedEdge.Begin() + start + length, EdgeSort());
            
            DEVICE_SYNC_CHECK();

            start += length;
        }
#endif

        DEVICE_SYNC_CHECK();

#if defined PROFILE
        sortTimer.Tock();
#endif
        uint edgeCount = 2 * m_elementCount;
        uint edgeBlock = edgeCount < 256 ? edgeCount : 256;
        uint edgeGrid = nutty::cuda::GetCudaGrid(edgeCount, edgeBlock);

        #if defined PROFILE
            cuTimer& reorderTimer = m_timer[timerIndex++];
            reorderTimer.Tick();
        #endif

        reorderEdges<<<edgeGrid, edgeBlock>>>
            (
            edgesDst, edgesSrc, edgeCount
            );

        #if defined PROFILE
            reorderTimer.Tock();
        #endif

        DEVICE_SYNC_CHECK();

        #if defined PROFILE
            cuTimer& splitTimer = m_timer[timerIndex++];
            splitTimer.Tick();
        #endif

        computeSAHSplits<<<edgeGrid, edgeBlock>>>
            (
            edgesDst, m_nodes, m_splits, m_nodesContent, d, edgeCount
            );

        #if defined PROFILE
            splitTimer.Tock();
        #endif

        DEVICE_SYNC_CHECK();

        start = 0;

#if defined PROFILE
        cuTimer& maxTimer = m_timer[timerIndex++];
        maxTimer.Tick();
#endif
#if defined DYNAMIC_PARALLELISM
        startGetMinSplits<<<nodeGrid, nodeBlock>>>
            (
            m_nodes, 
            m_splitsIndexedSplit.Begin()(),
            (1 << d) - 1
            );
#else
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
#endif

        
#if defined PROFILE
        maxTimer.Tock();
#endif

        /*reorderSplits<<<edgeGrid, edgeBlock>>> //dont need to reorder all
            (
            splitsDst, splitsSrc, edgeCount
            );*/
        
#if defined PROFILE
        cuTimer& setNodesTimer = m_timer[timerIndex++];
        setNodesTimer.Tick();
#endif

        setNodesSplitNAxis<<<nodeGrid, nodeBlock>>>
            (
            m_nodes, m_splits, d
            );

#if defined PROFILE
        setNodesTimer.Tock();
#endif

        DEVICE_SYNC_CHECK();
        #if defined PROFILE
        cuTimer& procEdgesTimer = m_timer[timerIndex++];
        procEdgesTimer.Tick();
        #endif

        classifyEdges<<<edgeGrid, edgeBlock>>>
            (
            m_nodes, edgesDst, m_edgeMask.Begin()(), d, edgeCount
            );

#if defined PROFILE
            procEdgesTimer.Tock();
#endif
        DEVICE_SYNC_CHECK();

#if defined PROFILE
            cuTimer& scanTimer = m_timer[timerIndex++];
            scanTimer.Tick();
#endif
        nutty::ExclusivePrefixSumScan(m_edgeMask.Begin(), m_edgeMask.Begin() + edgeCount + 1, m_scannedEdgeMask.Begin(), m_edgeMaskSums.Begin());

#if defined PROFILE
        scanTimer.Tock();
#endif
        DEVICE_SYNC_CHECK();

#if defined PROFILE
        cuTimer& initChilds = m_timer[timerIndex++];
        initChilds.Tick();
#endif

        /*DEBUG_OUT_A("%d\n", d);
        for(int i = 0; i < nodeCount; ++i)
        {
            uint os = (1 << d) - 1;
            uint below = m_nodesBelow[os + i];
            uint above = m_nodesAbove[os + i];
            uint prefix = m_nodesStartAdd[os + i];
            DEBUG_OUT_A("%d %d %d\n", below, above, prefix);
        } 

        DEBUG_OUT("-------------------------\n");*/

        initCurrentNodesAndCreateChilds<<<nodeGrid, nodeBlock>>>
            (
            m_scannedEdgeMask.Begin()(), m_edgeMask.Begin()(), m_nodes, d, d == (m_depth - 2)
            );

#if defined PROFILE
            initChilds.Tock();
#endif

        DEVICE_SYNC_CHECK();

#if defined PROFILE
            cuTimer& compactTimer = m_timer[timerIndex++];
            compactTimer.Tick();
#endif

       compactContentFromEdges<<<edgeGrid, edgeBlock>>>
            (
            edgesDst, m_nodes, m_nodesContent, m_edgeMask.Begin()(), m_scannedEdgeMask.Begin()(), d, edgeCount
            );

#if defined PROFILE
            compactTimer.Tock();
#endif

        DEVICE_SYNC_CHECK();

        PRINT_BUFFER(m_edgeMask);
        PRINT_BUFFER(m_scannedEdgeMask);
        PRINT_BUFFER(m_nodesContentCount);

       // DEBUG_OUT_A("m_elementCount=%d -> ", m_elementCount);
        uint lastCnt = m_elementCount;
        m_elementCount = m_scannedEdgeMask[edgeCount - 1] + m_edgeMask[edgeCount - 1];

        if(m_elementCount == 0)
        {
            m_elementCount = lastCnt;
            break;
        }

        //DEBUG_OUT_A("m_elementCount=%d\n", m_elementCount);
        if(2 * m_elementCount > m_edgeMask.Size() && d < m_depth-1)
        {
            GrowMemory();
        }

        DEVICE_SYNC_CHECK();
    }

    if(m_tprimitives.Size() < m_elementCount)
    {
        m_tprimitives.Resize(m_elementCount);
        m_tPrimAABBs.Resize(m_elementCount);
    }

    uint block = m_elementCount < 256 ? m_elementCount : 256;
    uint grid = nutty::cuda::GetCudaGrid(m_elementCount, block);

#if defined PROFILE
        cuTimer& pp = m_timer[timerIndex++];
        pp.Tick();
#endif

//     postprocess<<<grid, block>>>
//         (
//         m_primAABBs.Begin()(), m_tPrimAABBs.Begin()(), m_nodesContent, m_elementCount
//         );

    postprocess<<<grid, block>>>
        (
        m_primitives->Begin()(), m_tprimitives.Begin()(), m_nodesContent, m_elementCount
        );

    /*block = m_nodesContent < 256 ? m_nodesContent : 256;
    grid = nutty::cuda::GetCudaGrid((1 << m_depth) , block);

    postprocessNodes<<<grid, block>>>
    (
    m_nodes, m_depthFirstMask.Begin(), (1 << m_depth)
    );*/
    

    /*for(size_t i = 0; i < GetNodesCount(); ++i)
    {
        uint pos = i == 0 ? 0 : m_depthFirstMask[i-1];
//        DEBUG_OUT_A("%d ", pos);
        
        nutty::Copy(m_dfoNodesIsLeaf.Begin() + i, m_nodesIsLeaf.Begin() + i, 1);
        nutty::Copy(m_dfoNodesSplit.Begin() + i, m_nodesSplit.Begin() + i, 1);
        nutty::Copy(m_dfoNodesSplitAxis.Begin() + i, m_nodesSplitAxis.Begin() + i, 1);
        nutty::Copy(m_dfoNodesContentCount.Begin() + i, m_nodesContentCount.Begin() + i, 1);
        nutty::Copy(m_dfoNodesStartAdd.Begin() + i, m_nodesStartAdd.Begin() + i, 1);

        DEBUG_OUT_A("%d %d \n", i, pos);
    }*/

#if defined PROFILE
    pp.Tock();
#endif

    PRINT_BUFFER_N(m_nodesBBox);
    PRINT_BUFFER(m_primIndex);

#if defined PROFILE
        if(g_profilingSteps++ == 100)
        {
            std::ofstream out("profiling.txt");

            out << "perPrimBBox" << "\n";
            out << "initNodesContent" << "\n";
            out << "computeSceneBBox" << "\n";
            out << "createEdges" << "\n";
            out << "sortEdges" << "\n";
            out << "reorderEdges" << "\n";
            out << "computeSAHSplits" << "\n";
            out << "setNodesSplitNAxis" << "\n";
            out << "reduceSplits" << "\n";
            out << "cassifyEdges" << "\n";
            out << "scanEdges" << "\n";
            out << "initCurrentNodesNCreateChilds" << "\n";
            out << "compactContent" << "\n";
            out << "postProcess" << "\n";
 
            out << "\n" << m_timer[0].GetAverageMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[1].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[2].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[3].GetAverageMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[4].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[5].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[6].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[7].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[8].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[9].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[10].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[11].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[12].GetMillis()/(float)g_profilingSteps;
            out << "\n"  << m_timer[13].GetMillis()/(float)g_profilingSteps;    

            out.close();
            exit(0);
        }
#endif
}

template <typename T>
kdTree<T>::kdTree(byte depth, byte maxDepth) : 
    m_cudaModule(NULL), 
    m_depth(depth), 
    m_maxDepth(maxDepth), 
    m_nodesCount(0), 
    m_elementCount(0),
    m_sphereRadius(1)
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
    return &m_tprimitives;
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
nutty::DeviceBuffer<BBox>* kdTree<T>::GetPrimAABBs(void)
{
    return &m_tPrimAABBs;
}

template <typename T>
void kdTree<T>::SetDepth(uint d)
{
    m_depth = min(m_maxDepth, max(0, d));
}

template <typename T>
kdTree<T>::~kdTree(void)
{
    SAFE_DELETE(m_primitives);
    SAFE_DELETE(m_pStreamPool);
}

template <typename T>
void kdTree<T>::Init(void* prims, uint elements, PrimType type, float radius)
{
    m_elementCount = elements;

    m_sphereRadius = radius;

    m_primType = type;

    m_nodesCount = max(1, (1 << (uint)(m_maxDepth)) - 1);

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

extern "C" IKDTree* generate(nutty::DeviceBuffer<float3>* data, uint elements, uint d, uint maxDepth, PrimType type, float radius)
{
    g_tree = new kdTree<float3>(d, maxDepth);
    g_tree->Init((void*)data, elements, type, radius);
    //g_tree->Generate();

//     nutty::HostBuffer<uint> b(g_tree->GetNodesCount());
//     dfs(0, g_tree->GetNodesCount(), b);
// 
//     for(int i = 0; i < b.Size(); ++i)
//     {
//        // DEBUG_OUT_A("%d ", b[i]);
//     }

    /*
    const size_t h = 3;
    const int c = (1 << (h+1)) - 1;
    std::vector<int> veblayout;

    for(int i = 0; i < c; ++i)
    {
        veblayout.push_back(i);
    }

    VEB<int, h> veb(veblayout.data());
    int* add = (int*)&veb;
    for(int i = 0; i < c; ++i)
    {
        int addrDef = *(add+i);
        DEBUG_OUT_A("%d ", addrDef);
    }
    */
    return g_tree;
}
