#include "kdTree.cuh"
#include "../../Nutty/Nutty/Sort.h"
#include "../../Nutty/Nutty/Reduce.h"
#include "../../Nutty/Nutty/DevicePtr.h"

template<>
struct ShrdMemory<IndexedEdge>
{
    __device__ IndexedEdge* Ptr(void) 
    { 
        extern __device__ __shared__ IndexedEdge s_edge[];
        return s_edge;
    }
};

template<>
struct ShrdMemory<IndexedSplit>
{
    __device__ IndexedSplit* Ptr(void) 
    { 
        extern __device__ __shared__ IndexedSplit s_split[];
        return s_split;
    }
};

#if defined DYNAMIC_PARALLELISM

__global__ void startSort(Node nodes, IndexedEdge* edges, uint offset)
{
    uint id = GlobalId;
    nutty::DevicePtr<IndexedEdge>::size_type start = 2 * nodes.contentStart[offset + id];
    nutty::DevicePtr<IndexedEdge>::size_type length = 2 * nodes.contentCount[offset + id];
    if(length > 0)
    {
        nutty::DevicePtr<IndexedEdge> ptr_start(edges + start);
        nutty::DevicePtr<IndexedEdge> ptr_end(edges + start + length);
        nutty::Sort(ptr_start, ptr_end, EdgeSort());
    }
}

__global__ void startGetMinSplits(Node nodes, IndexedSplit* splits, uint offset)
{
    uint id = GlobalId;
    nutty::DevicePtr<IndexedSplit>::size_type start = 2 * nodes.contentStart[offset + id];
    nutty::DevicePtr<IndexedSplit>::size_type length = 2 * nodes.contentCount[offset + id];

    if(length > 0)
    {
        IndexedSplit neutralSplit;
        neutralSplit.index = 0;
        neutralSplit.sah = FLT_MAX;
        nutty::DevicePtr<IndexedSplit> ptr_start(splits + start);
        nutty::base::ReduceDP(ptr_start, ptr_start, length, ReduceIndexedSplit(), neutralSplit);
    }
}
#endif