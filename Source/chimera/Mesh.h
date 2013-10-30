#pragma once
#include "stdafx.h"
#include <vector>
#include "Cache.h"
#include "AxisAlignedBB.h"
namespace chimera 
{
    class Mesh : public IMesh
    {
        friend class ObjLoader;

    private:
        std::list<Face> m_faces;
        std::vector<IndexBufferInterval> m_indexIntervals;
        CMResource m_materials;
        FLOAT* m_vertices;
        UINT* m_indices;
        UINT m_indexCount;
        UINT m_vertexCount;
        UINT m_vertexStride;
        util::AxisAlignedBB m_aabb;
    public:

        Mesh(VOID) { }

        CMResource& VGetMaterials(VOID)
        {
            return m_materials;
        }

        VOID VAddIndexBufferInterval(UINT start, UINT count, UINT material)
        {
            IndexBufferInterval bi;
            bi.start = start;
            bi.count = count;
            bi.material = material;
            m_indexIntervals.push_back(bi);
        }

        UINT VGetIndexCount(VOID) CONST { return m_indexCount; }

        UINT VGetVertexCount(VOID) CONST { return m_vertexCount; }

        UINT VGetVertexStride(VOID) CONST { return m_vertexStride; }

        CONST FLOAT* VGetVertices(VOID) CONST { return m_vertices; }

        CONST std::list<chimera::Face>& VGetFaces(VOID) CONST { return m_faces; }

        util::AxisAlignedBB& VGetAABB(VOID) { return m_aabb; }

        CONST UINT* VGetIndices(VOID) CONST { return m_indices; }

        VOID VSetIndices(UINT* indices, UINT count) { this->m_indices = indices; m_indexCount = count; }

        VOID VSetVertices(FLOAT* vertices, UINT count, UINT stride) { this->m_vertices = vertices; m_vertexCount = count; m_vertexStride = stride; }

        std::vector<chimera::IndexBufferInterval>& VGetIndexBufferIntervals(VOID) { return m_indexIntervals; }

        ~Mesh(VOID) 
        {
            SAFE_ARRAY_DELETE(m_vertices);
            SAFE_ARRAY_DELETE(m_indices);
        }
    };
}

