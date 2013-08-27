#pragma once
#include "stdafx.h"
#include "Resources.h"
#include <vector>
#include "AxisAlignedBB.h"
namespace tbd 
{
    struct Triple 
    {
        UINT position;
        UINT texCoord;
        UINT normal;
        ULONG hash;
        UINT index;
        Triple() : position(0), texCoord(0), normal(0), hash(0), index(0) { }

        BOOL Triple::operator==(const Triple& t0)
        {
            return t0.position == position && t0.texCoord == texCoord && t0.normal == normal;
        }

        friend bool operator==(const Triple& t0, const Triple& t1)
        {
            return t0.position == t1.position && t0.texCoord == t1.texCoord && t0.normal == t1.normal;
        }

        friend bool operator<(const Triple& t0, const Triple& t1)
        {
            return t0.position < t1.position;
        }
    };

    struct Face 
    {
        std::vector<Triple> m_triples;
    };

    struct IndexBufferInterval 
    {
        UINT start;
        UINT count;
        UINT material;
        IndexBufferInterval(VOID) : start(0), count(0), material(0) {}
    };

    class Mesh : public tbd::ResHandle 
    {
        friend class ObjLoader;

    private:
        std::list<tbd::Face> m_faces;
        std::vector<tbd::IndexBufferInterval> m_indexIntervals;
        tbd::Resource m_materials;
        FLOAT* m_vertices;
        UINT* m_indices;
        UINT m_indexCount;
        UINT m_vertexCount;
        UINT m_vertexStride;
        util::AxisAlignedBB m_aabb;
    public:

        Mesh(VOID) { }

        tbd::Resource& GetMaterials(VOID)
        {
            return m_materials;
        }

        VOID AddIndexBufferInterval(UINT start, UINT count, UINT material);

        UINT GetIndexCount(VOID) CONST { return m_indexCount; }

        UINT GetVertexCount(VOID) CONST { return m_vertexCount; }

        UINT GetVertexStride(VOID) CONST { return m_vertexStride; }

        CONST FLOAT* GetVertices(VOID) CONST { return m_vertices; }

        CONST std::list<tbd::Face>& GetFaces(VOID) CONST { return m_faces; }

        util::AxisAlignedBB& GetAABB(VOID) { return m_aabb; }

        CONST UINT* GetIndices(VOID) CONST { return m_indices; }

        VOID SetIndices(UINT* indices, UINT count) { this->m_indices = indices; m_indexCount = count; }

        VOID SetVertices(FLOAT* vertices, UINT count, UINT stride) { this->m_vertices = vertices; m_vertexCount = count; m_vertexStride = stride; }

        std::vector<tbd::IndexBufferInterval>& GetIndexBufferIntervals(VOID) { return m_indexIntervals; }

        ~Mesh(VOID) 
        {
            if(m_vertices)
            {
                delete[] m_vertices;
            }
            if(m_indices)
            {
                delete[] m_indices;
            }
        }
    };
}

