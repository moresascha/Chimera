#pragma once
#include "kdtree.cuh"

#if 0
#define EXTC extern "C"
#else
#define EXTC
#endif

EXTC
__global__ void initLeafs(Split allSplits, Node nodes, uint count, uint depth);

EXTC
__global__ void setNodesCount(
uint* prefixNodesContentCount, 
Split allSplits, 
Node nodes, 
uint count, 
uint depth);

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

EXTC
__global__ void computePerPrimEdges(Edge edges, AABB* aabbs, uint N);

EXTC
__global__ void reOrderFromEdges(Edge edgesDst, Edge edgesSrc, uint N);

EXTC
__global__ void reOrderFromSAH(Split splitDst, Split splitSrc, uint N);

template <
    typename T
>
__global__ void postProcess(T* transformed, T* original, Edge edges, uint N)
{
    uint id = GlobalId;
    
    if(id >= N)
    {
        return;
    }

    transformed[id] = original[edges.primId[id]];
}

EXTC 
__global__ void spreadContent(
Node nodes, 
uint count, 
uint depth, 
uint* perThreadNodePos, 
uint* prefixScan);

EXTC 
__global__ void splitNodes(AABB* aabb, Node splits, uint depth);

EXTC 
__global__ void computeSplits(
AABB* aabb, 
Split splits, 
Node nodes,
Edge edges, 
uint count, 
uint depth,
uint* perThreadNodePos,
uint* prefixScan);