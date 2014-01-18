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
            bool m_created;
            uint m_elementCount;

            ID3D11Buffer* m_pBuffer;
            D3D11_BUFFER_DESC m_desc;
            D3D11_SUBRESOURCE_DATA m_data;
            D3D11_MAPPED_SUBRESOURCE m_mappedData;

        public:
            Buffer(void);

            void VCreate(void);

            D3D11_MAPPED_SUBRESOURCE* Map(void);

            uint VGetElementCount(void) const { return m_elementCount; }

            ID3D11Buffer* GetBuffer(void) { return m_pBuffer; }
            
            void* VGetDevicePtr(void)
            {
                return (void*)GetBuffer();
            }

            //WARNING: this might be NULL most of the time
            const void* GetRawData(void) { return m_data.pSysMem; }

            void DeleteRawData(void);

            void Unmap(void);

            void VSetData(const void* v, uint bytes);

            virtual ~Buffer(void);
        };

        class IndexBuffer : public Buffer
        {
        public:
            IndexBuffer(void);

            IndexBuffer(const uint* data, uint size);

            uint VGetByteCount(void) const { return m_elementCount; }

            void VBind(void);

            ~IndexBuffer(void);
        };

        class VertexBuffer : public Buffer, public IVertexBuffer
        {
        private:
            uint m_stride;
            uint m_offset;

        public:

            VertexBuffer(void);

            VertexBuffer(uint vertexCount, uint stride, const void* data = NULL, bool cpuWrite = false);

            void VBind(void);

            uint VGetStride(void) const { return m_stride; }

            void VInitParamater(uint vertexCount, uint stride, const void* data = NULL, bool cpuAccessFlags = false);

            uint VGetByteCount(void) const { return m_elementCount * m_stride; }

            uint VGetOffset(void) const { return m_offset; }

            ~VertexBuffer(void);
        };

        class Geometry;

        class GeometryDrawer
        {
        public:
            virtual void VDraw(Geometry* geo, const uint& count, const uint& start, const uint& vertexbase) {}
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
            bool m_initialized;        

            uint m_elementCount;

            GeometryDrawer* m_pDrawer;

            bool m_ownsInstanceBuffer;

        public:

            static void Create(void);

            static void Destroy(void);

            Geometry(bool ownsInstanceBuffer = false);

            //VRamRessource Interface
            virtual bool VCreate(void);

            virtual void VDestroy();

            virtual uint VGetByteCount(void) const;

            virtual void VBind(void);

            virtual void VDraw(uint start, uint count);

            virtual void VDraw(void);

            void VSetTopology(GeometryTopology top);

            void VSetVertexBuffer(const float* vertices, uint count, uint stride, bool cpuWrite = false);

            void VSetIndexBuffer(const uint* indices, uint size);

            IVertexBuffer* VGetVertexBuffer(void);

            IVertexBuffer* VGetInstanceBuffer(void);

            IDeviceBuffer* VGetIndexBuffer(void);

            void VAddInstanceBuffer(float* data, uint count, uint stride);

            void VSetInstanceBuffer(IVertexBuffer* buffer);

            void SetOwnsInstanceBuffer(bool owns);

            void DeleteRawData(void);

            virtual ~Geometry(void);
        };
    }
}