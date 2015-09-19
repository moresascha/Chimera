#pragma once
#include "stdafx.h"
#include <vector>
#include "Cache.h"
#include "AxisAlignedBB.h"
namespace chimera 
{
    class MeshSet : public IMeshSet
    {
        friend class ObjLoader;
    private:
        std::map<std::string, std::shared_ptr<IMesh>> m_meshes;

    public:
        uint VGetMeshCount(void)
        {
            return (uint)m_meshes.size();
        }

        IMesh* VGetMesh(uint i)
        {
            auto it = m_meshes.begin();
            uint _i = 0;
            while(_i < i) { ++it; ++_i; }
            return (*(it)).second.get();
        }

        IMesh* VGetMesh(const std::string& name)
        {
            if(m_meshes.find(name) != m_meshes.end())
            {
                return m_meshes[name].get();
            }
            LOG_CRITICAL_ERROR_A("Mesh '%s' not found!", name.c_str());
            return NULL;
        }

        MeshIterator VBegin(void) { return m_meshes.begin(); }

        MeshIterator VEnd(void) { return m_meshes.end(); }
    };

    class Mesh : public IMesh
    {
        friend class ObjLoader;

    private:
        std::list<Face> m_faces;
        std::vector<IndexBufferInterval> m_indexIntervals;
        CMResource m_materials;
        float* m_vertices;
        uint* m_indices;
        uint m_indexCount;
        uint m_vertexCount;
        uint m_vertexStride;
        util::AxisAlignedBB m_aabb;

    public:
        Mesh(void) { }

        CMResource& VGetMaterials(void)
        {
            return m_materials;
        }

        void VAddIndexBufferInterval(uint start, uint count, uint material, GeometryTopology topo)
        {
            IndexBufferInterval bi;
            bi.topo = topo;
            bi.start = start;
            bi.count = count;
            bi.material = material;
            m_indexIntervals.push_back(bi);
        }

        uint VGetIndexCount(void) const { return m_indexCount; }

        uint VGetVertexCount(void) const { return m_vertexCount; }

        uint VGetVertexStride(void) const { return m_vertexStride; }

        const float* VGetVertices(void) const { return m_vertices; }

        const std::list<chimera::Face>& VGetFaces(void) const { return m_faces; }

        util::AxisAlignedBB& VGetAABB(void) { return m_aabb; }

        const uint* VGetIndices(void) const { return m_indices; }

        void VSetIndices(uint* indices, uint count) { m_indices = indices; m_indexCount = count; }

        void VSetVertices(float* vertices, uint count, uint stride) { this->m_vertices = vertices; m_vertexCount = count; m_vertexStride = stride; }

        std::vector<IndexBufferInterval>& VGetIndexBufferIntervals(void) { return m_indexIntervals; }

        ~Mesh(void) 
        {
            SAFE_ARRAY_DELETE(m_vertices);
            SAFE_ARRAY_DELETE(m_indices);
        }
    };
}

