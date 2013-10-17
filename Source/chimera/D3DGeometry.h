#pragma once
#include "stdafx.h"
#include "D3DGraphics.h"
namespace chimera
{
    namespace d3d
    {
        class Buffer : public D3DResource
        {
        protected:
            BOOL m_created;
            UINT m_elementCount;
            UINT m_stride;
            UINT m_offset;

            ID3D11Buffer* m_pBuffer;
            D3D11_BUFFER_DESC m_desc;
            D3D11_SUBRESOURCE_DATA m_data;
            D3D11_MAPPED_SUBRESOURCE m_mappedData;

        public:
            Buffer(VOID);

            VOID virtual Bind(VOID) = 0;

            VOID Create(VOID);

            D3D11_MAPPED_SUBRESOURCE* Map(VOID);

            UINT GetStride(VOID) CONST { return m_stride; }

            UINT GetElementCount(VOID) CONST { return m_elementCount; }

            UINT GetOffset(VOID) CONST { return m_offset; }

            ID3D11Buffer* GetBuffer(VOID) { return m_pBuffer; }

            //WARNING: this might be NULL most of the time
            CONST VOID* GetRawData(VOID) { return m_data.pSysMem; }

            VOID DeleteRawData(VOID);

            VOID Unmap(VOID);

            virtual ~Buffer(VOID);
        };

        class IndexBuffer : public Buffer 
        {
        public:
            IndexBuffer(CONST UINT* data, UINT size);

            VOID Bind(VOID);

            ~IndexBuffer(VOID);
        };

        class VertexBuffer : public Buffer 
        {
        public:

            VertexBuffer(VOID);

            VertexBuffer(CONST VOID* data, UINT vertexCount, UINT stride, D3D11_CPU_ACCESS_FLAG cpuAccessFlags = (D3D11_CPU_ACCESS_FLAG)0);

            VOID Bind(VOID);

            VOID SetData(CONST VOID* data, UINT vertexCount, UINT stride, D3D11_CPU_ACCESS_FLAG cpuAccessFlags = (D3D11_CPU_ACCESS_FLAG)0);

            ~VertexBuffer(VOID);
        };

        class VertexBufferHandle : public chimera::IVRamHandle
        {
        private:
            VertexBuffer* m_pVertexBuffer;
        public:
            VertexBufferHandle(VOID);
            VOID SetVertexData(CONST VOID* data, UINT vertexCount, UINT stride, D3D11_CPU_ACCESS_FLAG cpuAccessFlags = (D3D11_CPU_ACCESS_FLAG)0);
            BOOL VCreate(VOID);
            VOID VDestroy();
            UINT VGetByteCount(VOID) CONST;
            VertexBuffer* GetBuffer(VOID) CONST;
        };

        class Geometry;

        class GeometryDrawer
        {
        public:
            virtual VOID VDraw(Geometry* geo, CONST UINT& count, CONST UINT& start, CONST UINT& vertexbase) {}
        };

        class Geometry : public IGeometry
        {
            friend class VertexBuffer;
        protected:

            static Geometry* m_psCurrentBound;
            static GeometryDrawer* INSTANCED_DRAWER;
            static GeometryDrawer* DEFAULT_DRAWER;
            static GeometryDrawer* ARRAY_DRAWER;

            VertexBuffer* m_pVertexBuffer;
            IndexBuffer* m_pIndexBuffer;
            VertexBuffer* m_pInstanceBuffer;

            D3D_PRIMITIVE_TOPOLOGY m_primType;
            BOOL m_initialized;        

            UINT m_elementCount;

            GeometryDrawer* m_pDrawer;

            BOOL m_ownsInstanceBuffer;

        public:

            static VOID Create(VOID);

            static VOID Destroy(VOID);

            Geometry(BOOL ownsInstanceBuffer = FALSE);

            //VRamRessource Interface
            virtual BOOL VCreate(VOID);

            virtual VOID VDestroy();

            virtual UINT VGetByteCount(VOID) CONST;

            virtual VOID VBind(VOID);

            virtual VOID VDraw(UINT start, UINT count);

            virtual VOID VDraw(VOID);

            VOID VSetTopology(GeometryTopology top);

            VOID VSetVertexBuffer(CONST FLOAT* vertices, UINT count, UINT stride);

            VOID VSetIndexBuffer(CONST UINT* indices, UINT size);

            VertexBuffer* GetVertexBuffer(VOID);

            VertexBuffer* GetInstanceBuffer(VOID);

            IndexBuffer* GetIndexBuffer(VOID);

            VOID VAddInstanceBuffer(FLOAT* data, UINT count, UINT stride);

            VOID SetInstanceBuffer(VertexBuffer* buffer);

            VOID SetOwnsInstanceBuffer(BOOL owns);

            VOID DeleteRawData(VOID);

            virtual ~Geometry(VOID);
        };
    }
}