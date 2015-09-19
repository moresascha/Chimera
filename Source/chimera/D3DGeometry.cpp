#include "D3DGeometry.h"

namespace chimera
{
    namespace d3d
    {
        Geometry* Geometry::m_psCurrentBound = NULL;
        GeometryDrawer* Geometry::INSTANCED_DRAWER = NULL;
        GeometryDrawer* Geometry::DEFAULT_DRAWER = NULL;
        GeometryDrawer* Geometry::ARRAY_DRAWER = NULL;

        class InstancedDrawer : public GeometryDrawer
        {
        public:
            void VDraw(Geometry* geo, const uint& count, const uint& start, const uint& vertexbase)
            {
                chimera::d3d::GetContext()->DrawIndexedInstanced(count, geo->VGetInstanceBuffer(0)->VGetElementCount(), start, vertexbase, 0);
            }
        };

        class DefaultDrawer : public GeometryDrawer
        {
        public:
            void VDraw(Geometry* geo, const uint& count, const uint& start, const uint& vertexbase)
            {
                chimera::d3d::GetContext()->DrawIndexed(count, start, vertexbase);            
            }
        };

        class ArrayDrawer : public GeometryDrawer
        {
        public:
            void VDraw(Geometry* geo, const uint& count, const uint& start, const uint& vertexbase)
            {
                chimera::d3d::GetContext()->Draw(count, start);
            }
        };

        void Geometry::Create(void)
        {
            Geometry::INSTANCED_DRAWER = new InstancedDrawer();
            Geometry::DEFAULT_DRAWER = new DefaultDrawer();
            Geometry::ARRAY_DRAWER = new ArrayDrawer();
        }

        void Geometry::Destroy(void)
        {
            SAFE_DELETE(Geometry::INSTANCED_DRAWER);
            SAFE_DELETE(Geometry::DEFAULT_DRAWER);
            SAFE_DELETE(Geometry::ARRAY_DRAWER);
        }

        Buffer::Buffer(void) : m_pBuffer(0), m_created(false), m_elementCount(0)
        {
            ZeroMemory(&m_desc, sizeof(D3D11_BUFFER_DESC));
            ZeroMemory(&m_data, sizeof(D3D11_SUBRESOURCE_DATA));
        }

        void Buffer::VCreate(void) 
        {
            if(m_created) return;
            m_created = true;
            m_desc.Usage = GetUsage();
            m_desc.CPUAccessFlags = GetCPUAccess();
            m_desc.BindFlags = GetBindflags();
            D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateBuffer(&m_desc, m_data.pSysMem != NULL ? &m_data : NULL, &m_pBuffer));
        }

        D3D11_MAPPED_SUBRESOURCE* Buffer::Map(void)
        {
            D3D_SAVE_CALL(chimera::d3d::GetContext()->Map(m_pBuffer, 0, GetCPUAccess() == D3D11_CPU_ACCESS_WRITE ? D3D11_MAP_WRITE_DISCARD : D3D11_MAP_READ, 0, &m_mappedData));
            return &m_mappedData;
        }

        void Buffer::VSetData(const void* v, uint bytes)
        {
            D3D11_MAPPED_SUBRESOURCE* sub = Map();
            memcpy(sub->pData, v, bytes);
            Unmap();
        }

        void Buffer::Unmap(void)
        {
            chimera::d3d::GetContext()->Unmap(m_pBuffer, 0);
        }

        void Buffer::DeleteRawData(void)
        {
            SAFE_ARRAY_DELETE(m_data.pSysMem);
        }

        Buffer::~Buffer(void) 
        {
            SAFE_RELEASE(m_pBuffer);
        }

        IndexBuffer::IndexBuffer(void) { }

        IndexBuffer::IndexBuffer(const uint* data, uint size) 
        {
            SetBindflags(D3D11_BIND_INDEX_BUFFER);
            SetUsage(D3D11_USAGE_DEFAULT);
            m_desc.ByteWidth = sizeof(uint) * size;

            m_data.pSysMem = data;
            m_data.SysMemPitch = 0;
            m_data.SysMemSlicePitch = 0;

            m_elementCount = size;
        }

        void IndexBuffer::VBind(void) 
        {
            chimera::d3d::GetContext()->IASetIndexBuffer(m_pBuffer, DXGI_FORMAT_R32_UINT, 0);
        }

        IndexBuffer::~IndexBuffer() {}

        VertexBuffer::VertexBuffer(void) : m_stride(1), m_offset(0) { }

        VertexBuffer::VertexBuffer(uint vertexCount, uint stride, const void* data, bool cpuAccessFlags)
        {
            VInitParamater(vertexCount, stride, data, cpuAccessFlags);
        }

        void VertexBuffer::VInitParamater(uint vertexCount, uint stride, const void* data /* = NULL */, bool cpuAccessFlags /* = FALSE */)
        {
            m_offset = 0;
            m_stride = stride; 
            m_elementCount = vertexCount;
            if(cpuAccessFlags)
            {
                SetCPUAccess(D3D11_CPU_ACCESS_WRITE);
                SetUsage(D3D11_USAGE_DYNAMIC);
            }

            SetBindflags(D3D11_BIND_VERTEX_BUFFER);

            m_desc.ByteWidth = vertexCount * stride;

            m_data.pSysMem = data;
            m_data.SysMemPitch = 0;
            m_data.SysMemSlicePitch = 0;            
        }

        void VertexBuffer::VBind(void)
        {
            chimera::d3d::GetContext()->IASetVertexBuffers(0, 1, &m_pBuffer, &m_stride, &m_offset);
        }

        VertexBuffer::~VertexBuffer() {}

        Geometry::Geometry(bool ownsInstanceBuffer /* = false */) 
            : m_pVertexBuffer(NULL), m_pIndexBuffer(NULL), m_elementCount(0), m_initialized(false), m_primType(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST)
        {
            m_pDrawer = DEFAULT_DRAWER;

            for(int i = 0; i < 16; ++i)
            {
                m_ownsInstanceBufferMask[i] = false;
                m_pInstanceBuffer[i] = NULL;
            }
        }

        void Geometry::VSetIndexBuffer(const uint* indices, uint size) 
        {
            m_pIndexBuffer = new chimera::d3d::IndexBuffer(indices, size);
        }

        void Geometry::VSetVertexBuffer(const float* vertices, uint count, uint stride, bool cpuWrite) 
        {
            m_pVertexBuffer = new chimera::d3d::VertexBuffer(count, stride, vertices, cpuWrite);
        }

        void Geometry::VSetTopology(GeometryTopology primType)
        {
            if(primType == eTopo_Lines)
            {
                m_primType = D3D11_PRIMITIVE_TOPOLOGY_LINELIST;
            }
            else if(primType == eTopo_LineStrip)
            {
                m_primType = D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP;
            }
            else if(primType == eTopo_Triangles)
            {
                m_primType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            }
            else if(primType == eTopo_TriangleStrip)
            {
                m_primType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
            }
        }

        uint Geometry::VGetByteCount(void) const
        {
            uint indexBytes = 0;
            uint instancedBytes = 0;
            if(m_pIndexBuffer)
            {
                indexBytes = sizeof(uint) * m_pIndexBuffer->VGetElementCount();
            }
            for(int i = 0; i < 16; ++i)
            {
                IVertexBuffer* ib = m_pInstanceBuffer[i];
                if(ib)
                {
                    instancedBytes = ib->VGetElementCount() * ib->VGetStride() * sizeof(float);
                }
            }
            return m_pVertexBuffer->VGetElementCount() * m_pVertexBuffer->VGetStride() * sizeof(float) + instancedBytes + indexBytes;
        }

        IVertexBuffer* Geometry::VGetVertexBuffer(void)
        {
            return m_pVertexBuffer;
        }

        bool Geometry::VCreate(void)
        {
           // if(VIsReady()) return TRUE; //TODO needed?

            m_pVertexBuffer->VCreate();
        
            m_elementCount = m_pVertexBuffer->VGetElementCount();

            if(m_pIndexBuffer)
            {
                m_pDrawer = DEFAULT_DRAWER;
                m_pIndexBuffer->VCreate();
                m_elementCount = m_pIndexBuffer->VGetElementCount();
            }
            else
            {
                m_pDrawer = ARRAY_DRAWER;
            }

            if(m_pInstanceBuffer[0])
            {
                for(int i = 0; i < 16; ++i)
                {
                    IVertexBuffer* ib = m_pInstanceBuffer[i];
                    if(ib)
                    {
                        ib->VCreate();
                    }
                }
                m_pDrawer = INSTANCED_DRAWER;
            }

            return true;
        }

        void Geometry::VDraw(void)
        {
            m_pDrawer->VDraw(this, m_elementCount, 0, 0);
        }

        void Geometry::VDraw(uint start, uint count)
        {
            m_pDrawer->VDraw(this, count, start, 0);
        }

        void Geometry::VBind(void)
        {
            if(m_psCurrentBound != this)
            {
                if(m_pIndexBuffer)
                {
                    m_pIndexBuffer->VBind();
                }

                if(m_pInstanceBuffer)
                {
                    ID3D11Buffer* buffer[16];
                    uint strides[16];
                    uint offsets[16];
                    uint count = 1;

                    buffer[0] = (ID3D11Buffer*)m_pVertexBuffer->VGetDevicePtr();
                    strides[0] = m_pVertexBuffer->VGetStride();
                    offsets[0] = m_pVertexBuffer->VGetOffset();

                    for(int i = 1; i < 17; ++i)
                    {
                        IVertexBuffer* ib = m_pInstanceBuffer[i-1];
                        if(ib)
                        {
                            buffer[i] = (ID3D11Buffer*)ib->VGetDevicePtr();
                            strides[i] = ib->VGetStride();
                            offsets[i] = ib->VGetOffset();
                            count++;
                        }
                    }

                    chimera::d3d::GetContext()->IASetVertexBuffers(0, count, buffer, strides, offsets);
                }
                else
                {
                    m_pVertexBuffer->VBind();
                }

                m_psCurrentBound = this;
                
                chimera::d3d::GetContext()->IASetPrimitiveTopology(m_primType);
            }
        }

        void Geometry::VDestroy(void)
        {
            SAFE_DELETE(m_pIndexBuffer);
            SAFE_DELETE(m_pVertexBuffer);
            for(int i = 0; i < 16; ++i)
            {
                IVertexBuffer* ib = m_pInstanceBuffer[i];
                bool mask = m_ownsInstanceBufferMask[i];
                if(ib && mask)
                {
                    SAFE_DELETE(ib);
                }
            }
        }

        IVertexBuffer* Geometry::VGetInstanceBuffer(uint index)
        {
            return m_pInstanceBuffer[index];
        }

        IDeviceBuffer* Geometry::VGetIndexBuffer(void)
        {
            return m_pIndexBuffer;
        }

        void Geometry::SetOwnsInstanceBuffer(bool owns, uint index)
        {
            m_ownsInstanceBufferMask[index] = owns;
        }

//         VOID Geometry::VAddInstanceBuffer(FLOAT* data, UINT count, UINT stride)
//         {
//             if(!m_ownsInstanceBuffer)
//             {
//                 LOG_CRITICAL_ERROR("Geometry can't own an instance buffer!");
//             }
//             m_pInstanceBuffer = new chimera::d3d::VertexBuffer(data, count, stride);
//             m_pDrawer = INSTANCED_DRAWER;
//         }

        void Geometry::VSetInstanceBuffer(IVertexBuffer* buffer, uint slot)
        {
            m_pInstanceBuffer[slot] = buffer;

            assert(m_ownsInstanceBufferMask[slot] == false);

            m_ownsInstanceBufferMask[slot] = false;

            if(m_psCurrentBound == this)
            {
                m_psCurrentBound = NULL;
            }

            m_pDrawer = buffer != NULL ? INSTANCED_DRAWER : DEFAULT_DRAWER;
        }

        void Geometry::DeleteRawData(void)
        {
            if(m_pIndexBuffer)
            {
                m_pIndexBuffer->DeleteRawData();
            }
            if(m_pVertexBuffer)
            {
                m_pVertexBuffer->DeleteRawData();
            }
        }

        Geometry::~Geometry(void) 
        {
            VDestroy();
        }
    }
}