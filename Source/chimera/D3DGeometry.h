#pragma once
#include "stdafx.h"
#include "D3DGraphics.h"
namespace chimera
{
    namespace d3d
    {
        class Buffer : public D3DResource, public virtual IDeviceBuffer
        {
        protected:
            BOOL m_created;
            UINT m_elementCount;

            ID3D11Buffer* m_pBuffer;
            D3D11_BUFFER_DESC m_desc;
            D3D11_SUBRESOURCE_DATA m_data;
            D3D11_MAPPED_SUBRESOURCE m_mappedData;

        public:
            Buffer(VOID);

            VOID VCreate(VOID);

            D3D11_MAPPED_SUBRESOURCE* Map(VOID);

            UINT VGetElementCount(VOID) CONST { return m_elementCount; }

            ID3D11Buffer* GetBuffer(VOID) { return m_pBuffer; }
            
            VOID* VGetDevicePtr(VOID)
            {
                return (VOID*)GetBuffer();
            }

            //WARNING: this might be NULL most of the time
            CONST VOID* GetRawData(VOID) { return m_data.pSysMem; }

            VOID DeleteRawData(VOID);

            VOID Unmap(VOID);

            VOID VSetData(CONST VOID* v, UINT bytes);

            virtual ~Buffer(VOID);
        };

        class IndexBuffer : public Buffer
        {
        public:
            IndexBuffer(VOID);

            IndexBuffer(CONST UINT* data, UINT size);

            UINT VGetByteCount(VOID) CONST { return m_elementCount; }

            VOID VBind(VOID);

            ~IndexBuffer(VOID);
        };

        class VertexBuffer : public Buffer, public IVertexBuffer
        {
        private:
            UINT m_stride;
            UINT m_offset;

        public:

            VertexBuffer(VOID);

            VertexBuffer(UINT vertexCount, UINT stride, CONST VOID* data = NULL, BOOL cpuWrite = FALSE);

            VOID VBind(VOID);

            UINT VGetStride(VOID) CONST { return m_stride; }

            VOID VInitParamater(UINT vertexCount, UINT stride, CONST VOID* data = NULL, BOOL cpuAccessFlags = FALSE);

            UINT VGetByteCount(VOID) CONST { return m_elementCount * m_stride; }

            UINT VGetOffset(VOID) CONST { return m_offset; }

            ~VertexBuffer(VOID);
        };

        class Geometry;

        class GeometryDrawer
        {
        public:
            virtual VOID VDraw(Geometry* geo, CONST UINT& count, CONST UINT& start, CONST UINT& vertexbase) {}
        };

        class Geometry : public IGeometry
        {
        protected:

            static Geometry* m_psCurrentBound;
            static GeometryDrawer* INSTANCED_DRAWER;
            static GeometryDrawer* DEFAULT_DRAWER;
            static GeometryDrawer* ARRAY_DRAWER;

            VertexBuffer* m_pVertexBuffer;
            IVertexBuffer* m_pInstanceBuffer;

            IndexBuffer* m_pIndexBuffer;

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

            VOID VSetVertexBuffer(CONST FLOAT* vertices, UINT count, UINT stride, BOOL cpuWrite = FALSE);

            VOID VSetIndexBuffer(CONST UINT* indices, UINT size);

            IVertexBuffer* VGetVertexBuffer(VOID);

            IVertexBuffer* VGetInstanceBuffer(VOID);

            IDeviceBuffer* VGetIndexBuffer(VOID);

            VOID VAddInstanceBuffer(FLOAT* data, UINT count, UINT stride);

            VOID VSetInstanceBuffer(IVertexBuffer* buffer);

            VOID SetOwnsInstanceBuffer(BOOL owns);

            VOID DeleteRawData(VOID);

            virtual ~Geometry(VOID);
        };
    }
}