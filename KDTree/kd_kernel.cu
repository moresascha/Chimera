#pragma once
#include "kd_kernel.cuh"

struct DstDesc
{
    int isright;
    uint prefixSum;
    uint below;
    float s;
};

//wahrscheinlich überflüssig: todo
__global__ void initLeafs(Split allSplits, Node nodes, uint count, uint depth)
{
    uint id = GlobalId;

	uint contentCountOffsetMe = elemsBeforeLevel(depth);
    uint sa = 0;
    int last = -1;

    for(uint i = 0; i < id; ++i)
    {
        last = nodes.contentCount[contentCountOffsetMe + i];
        sa += last;
    }

    Split_v1 s;
    s.split = 0;
    s.below = s.above = 0;
    s.primId = 0; //todo
    s.sah.sah = 0;
    s.sah.index = 0;
    s.axis = 0;

    if(sa < count)
    {
        s.split = allSplits.split[sa];
        s.below = allSplits.below[sa];
        s.above = allSplits.above[sa];
        s.primId = allSplits.primId[sa];
        s.sah = allSplits.sah[sa];
        s.axis = allSplits.axis[sa];
    }    

 	//nodes.contentCount[contentCountOffsetMe + id] = (id%2) * s.above + ((id+1)%2) * s.below;
    nodes.contentStartIndex[contentCountOffsetMe + id] = sa;
    nodes.split[contentCountOffsetMe + id] = 0;
    nodes.leaf[contentCountOffsetMe + id] = 1;
    nodes.axis[contentCountOffsetMe + id] = 0;
    nodes.below[contentCountOffsetMe + id] = s.below;
}

__global__ void setNodesCount(
    uint* prefixNodesContentCount, 
    Split allSplits, 
    Node nodes, 
    uint count, 
    uint depth)
{
    uint id = GlobalId;

    uint contentCountOffset = elemsBeforeNextLevel(depth);
	uint contentCountOffsetMe = elemsBeforeLevel(depth);
    uint sa = 0;
    int last = -1;

    for(uint i = 0; i < id; ++i)
    {
        last = nodes.contentCount[contentCountOffsetMe + i];
        sa += last;
    }

    Split_v1 s;
    s.split = 0;
    s.below = s.above = 0;
    s.primId = 0;
    s.sah.sah = 0;
    s.sah.index = 0;

    if(sa < count)
    {
        s.split = allSplits.split[sa];
        s.below = allSplits.below[sa];
        s.above = allSplits.above[sa];
        s.primId = allSplits.primId[sa];
        s.sah = allSplits.sah[sa];
        s.axis = allSplits.axis[sa];
    }  

    if(last > 0 || sa == 0)
    {
        nodes.contentCount[contentCountOffset + 2 * id + 0] = s.below;
        nodes.contentCount[contentCountOffset + 2 * id + 1] = s.above;
    }

    //nodes.contentCount[contentCountOffsetMe + id] = (s.below + s.above);
    nodes.contentStartIndex[contentCountOffsetMe + id] = sa;
    nodes.split[contentCountOffsetMe + id] = s.split;
    nodes.leaf[contentCountOffsetMe + id] = (s.below + s.above) < 2;
    nodes.axis[contentCountOffsetMe + id] = s.axis;
    nodes.below[contentCountOffsetMe + id] = s.below;
}

/*__global__ void computePerPrimEdges(Edge edges, AABB* aabbs, uint N)
{
    uint id = GlobalId;
    if(id >= N)
    {
        return;
    }

    AABB aabb = aabbs[id];

    Edge_v1 start;
    start.primId = id;
    start.indexedEdge.index = 2 * id;
    start.type = eStart;
    Edge_v1 end;
    end.type = eEnd;
    end.indexedEdge.index = 2 * id + 1;
    end.primId = id;

    /* for(byte i = 0; i < 3; ++i)
    {
        setEdgeSplit(&start, i, getAxis(&aabb.min, i));
        setEdgeSplit(&end, i, getAxis(&aabb.max, i));
    }* /

    todo, start.t3 = aabb.min;
    todo, end.t3 = aabb.max;

    edges.type[2 * id + 0] = start.type;
    edges.indexedEdge[2 * id + 0] = start.indexedEdge;
    edges.primId[2 * id + 0] = start.primId;

    edges.type[2 * id + 1] = end.type;
    edges.indexedEdge[2 * id + 1] = end.indexedEdge;
    edges.primId[2 * id + 1] = end.primId;
}*/

__global__ void computePerPrimEdges(Edge edges, AABB* aabbs, uint N)
{
    const uint id = GlobalId;
    if(id >= N)
    {
        return;
    }
    const int edgesCount = 8;

    AABB aabb = aabbs[id];

    Edge_v1 start;
    start.primId = id;
    //start.indexedEdge.t3 = aabb.min;

    Edge_v1 end;
    end.primId = id;
    //end.indexedEdge.t3 = aabb.max;

    byte index = 0;

    for(byte ix = 0; ix < 2; ++ix)
    {
        float x = aabb.getX(ix);
        for(byte iy = 0; iy < 2; ++iy)
        {
            float y = aabb.getY(iy);

            float z_min = aabb.getZ(0);
            float z_max = aabb.getZ(1);

            start.indexedEdge.t[0].type = (EdgeType)ix;
            start.indexedEdge.t[0].v = x;
            
            start.indexedEdge.t[1].type = (EdgeType)iy;
            start.indexedEdge.t[1].v = y;

            start.indexedEdge.t[2].type = eStart;
            start.indexedEdge.t[2].v = z_min;

            start.indexedEdge.index = edgesCount * id + index;

            edges.indexedEdge[edgesCount * id + index] = start.indexedEdge;
            edges.primId[edgesCount * id + index] = start.primId;
            index++;

            end.indexedEdge.t[0].type = (EdgeType)ix;
            end.indexedEdge.t[0].v = x;
            
            end.indexedEdge.t[1].type = (EdgeType)iy;
            end.indexedEdge.t[1].v = y;

            end.indexedEdge.t[2].type = eEnd;
            end.indexedEdge.t[2].v = z_max;

            end.indexedEdge.index = edgesCount * id + index;

            edges.indexedEdge[edgesCount * id + index] = end.indexedEdge;
            edges.primId[edgesCount * id + index] = end.primId;
            index++;
        }
    }
}

__global__ void reOrderFromEdges(Edge edgesDst, Edge edgesSrc, uint N)
{
    uint id = GlobalId;
    if(id >= N)
    {
        return;
    }
    uint src = edgesSrc.indexedEdge[id].index;
    edgesDst.primId[id] = edgesSrc.primId[src];
    /*edgesDst.indexedEdge[id].t[0] = edgesSrc.indexedEdge[src].t[0];
    edgesDst.indexedEdge[id].t[1] = edgesSrc.indexedEdge[src].t[1];
    edgesDst.indexedEdge[id].t[2] = edgesSrc.indexedEdge[src].t[2];*/
    edgesDst.indexedEdge[id].index = id;
}

__global__ void reOrderFromSAH(Split splitDst, Split splitSrc, uint N)
{
    uint id = GlobalId;
    if(id >= N)
    {
        return;
    }
    uint src = splitSrc.sah[id].index;
    splitDst.primId[id] = splitSrc.primId[src];
    splitDst.axis[id] = splitSrc.axis[src];
    splitDst.split[id] = splitSrc.split[src];
    splitDst.below[id] = splitSrc.below[src];
    splitDst.above[id] = splitSrc.above[src];
    splitDst.sah[id].index = id;
}

__device__ DstDesc getDstNode(Node nodes, uint* perThreadNodePos, uint* prefixScan, uint depth)
{
    uint id = GlobalId;
    
    uint startAdd = elemsBeforeLevel(depth);

    uint index = perThreadNodePos[id];

    float s = nodes.split[startAdd + index];
    uint below = nodes.below[startAdd + index];
    
    uint prefixSum = prefixScan[id];
    
    uint modi = id - prefixSum;
    
    int right = modi >= below;

    DstDesc d;
    d.isright = right;
    d.below = below;
    d.prefixSum = prefixSum;
    d.s = s;

    return d;
}

__device__ uint getNextNodePos(uint* perThreadNodePos, int isright, uint depth)
{
    uint id = GlobalId;

    uint scanVal = perThreadNodePos[id];

    scanVal = 2 * scanVal;

    return scanVal + isright;
}

EXTC __global__ void spreadContent(
    Node nodes,
    uint count, 
    uint depth, 
    uint* perThreadNodePos, 
    uint* prefixScan)
{
    uint id = GlobalId;

    if(id >= count)
    {
        return;
    }

    DstDesc dd = getDstNode(nodes, perThreadNodePos, prefixScan, depth);

    uint nextScanVal = getNextNodePos(perThreadNodePos, dd.isright, depth);

    if(!dd.isright)
    {
        prefixScan[id] = dd.prefixSum;
    }
    else
    {
        prefixScan[id] = dd.prefixSum + dd.below;
    }

    perThreadNodePos[id] = nextScanVal;
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

void __device__ splitAABB(AABB* aabb, float split, byte axis, AABB* l, AABB* r)
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

EXTC __global__ void splitNodes(AABB* aabb, Node nodes, uint depth)
{
    uint pos = (1 << (depth+1)) - 1 + 2*GlobalId;
    uint me = (1 << (depth)) - 1 + GlobalId;

    AABB nodeAABB = aabb[me];
    AABB l; AABB r;

    r.min = make_float3(0,0,0);
    r.max = make_float3(0,0,0);
    l.min = make_float3(0,0,0);
    l.max = make_float3(0,0,0);
    
    float split = nodes.split[me];
    byte axis = nodes.axis[me];

    splitAABB(&nodeAABB, split, axis, &l, &r);

    aabb[pos + 0] = l;
    aabb[pos + 1] = r;
}

EXTC __global__ void computeSplits(
    AABB* aabb, 
    Split splits, 
    Node nodes,
    Edge edges, 
    uint count, 
    uint depth,
    uint* perThreadNodePos,
    uint* prefixScan)
{
    uint id = GlobalId;
    
    if(id >= count)
    {
        return;
    }

    uint levelOffset = elemsBeforeLevel(depth);

    uint pos = id - prefixScan[id];

    uint nodeIndex = perThreadNodePos[id];

    AABB aa = aabb[levelOffset + nodeIndex];
    
    int axis = getLongestAxis(aa.min, aa.max);

    uint elemsInNode = nodes.contentCount[levelOffset + nodeIndex];
    EdgeType type = edges.indexedEdge[id].t[axis].type;
    float split = getSplit(edges, id, axis);
    splits.split[id] = split;
    splits.axis[id] = axis;
    uint above = max((int)(elemsInNode - pos - (type == eStart ? 0 : 1)), 0);
    uint below = max((int)(pos + (type == eEnd ? 1 : 0)), 0);
    splits.above[id] = above;
    splits.below[id] = below;
    splits.primId[id] = edges.primId[id];

    float s = getSAH(&aa, axis, split, pos, elemsInNode - pos, !(above || below));
    IndexedSAH sah;
    sah.sah = s;
    sah.index = id;
    splits.sah[id] = sah;
}
