#include "kdtree.cuh"
#include "../../Nutty/Nutty/Reduce.h"
#include "../../Nutty/Nutty/Sort.h"

__global__ void DP_computeInitialBBox(AABB* dst, AABB* src, float3* data)
{
	//nutty::Reduce(m_aabbMin.Begin(), dataBuffer, dataBuffer + elementCount, float3min());

    //nutty::Reduce(m_aabbMax.Begin(), dataBuffer, dataBuffer + elementCount, float3max());
}
