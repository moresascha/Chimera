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
            VOID VDraw(Geometry* geo, CONST UINT& count, CONST UINT& start, CONST UINT& vertexbase)
            {
                chimera::d3d::GetContext()->DrawIndexedInstanced(count, geo->GetInstanceBuffer()->GetElementCount(), start, vertexbase, 0);
            }
        };

        class DefaultDrawer : public GeometryDrawer
        {
        public:
            VOID VDraw(Geometry* geo, CONST UINT& count, CONST UINT& start, CONST UINT& vertexbase)
            {
                chimera::d3d::GetContext()->DrawIndexed(count, start, vertexbase);            
            }
        };

        class ArrayDrawer : public GeometryDrawer
        {
        public:
            VOID VDraw(Geometry* geo, CONST UINT& count, CONST UINT& start, CONST UINT& vertexbase)
            {
                chimera::d3d::GetContext()->Draw(count, start);
            }
        };

        VOID Geometry::Create(VOID)
        {
            Geometry::INSTANCED_DRAWER = new InstancedDrawer();
            Geometry::DEFAULT_DRAWER = new DefaultDrawer();
            Geometry::ARRAY_DRAWER = new ArrayDrawer();
        }

        VOID Geometry::Destroy(VOID)
        {
            SAFE_DELETE(Geometry::INSTANCED_DRAWER);
            SAFE_DELETE(Geometry::DEFAULT_DRAWER);
            SAFE_DELETE(Geometry::ARRAY_DRAWER);
        }

        Buffer::Buffer(VOID) : m_pBuffer(0), m_created(FALSE), m_elementCount(0), m_stride(1), m_offset(0)
        {
            ZeroMemory(&m_desc, sizeof(D3D11_BUFFER_DESC));
            ZeroMemory(&m_data, sizeof(D3D11_SUBRESOURCE_DATA));
        }

        VOID Buffer::Create(VOID) 
        {
            if(m_created) return;
            m_created = TRUE;
            m_desc.Usage = GetUsage();
            m_desc.CPUAccessFlags = GetCPUAccess();
            m_desc.BindFlags = GetBindflags();
            D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateBuffer(&m_desc, &m_data, &m_pBuffer));
        }

        D3D11_MAPPED_SUBRESOURCE* Buffer::Map(VOID)
        {
            D3D_SAVE_CALL(chimera::d3d::GetContext()->Map(m_pBuffer, 0, GetCPUAccess() == D3D11_CPU_ACCESS_WRITE ? D3D11_MAP_WRITE_DISCARD : D3D11_MAP_READ, 0, &m_mappedData));
            return &m_mappedData;
        }

        VOID Buffer::VSetData(VOID* v, UINT bytes)
        {
            D3D11_MAPPED_SUBRESOURCE* sub = Map();
            memcpy(sub->pData, v, bytes);
            Unmap();
        }

        VOID Buffer::Unmap(VOID)
        {
            chimera::d3d::GetContext()->Unmap(m_pBuffer, 0);
        }

        VOID Buffer::DeleteRawData(VOID)
        {
            SAFE_ARRAY_DELETE(m_data.pSysMem);
        }

        Buffer::~Buffer(VOID) 
        {
            SAFE_RELEASE(m_pBuffer);
        }

        IndexBuffer::IndexBuffer(CONST UINT* data, UINT size) 
        {
            SetBindflags(D3D11_BIND_INDEX_BUFFER);
            SetUsage(D3D11_USAGE_DEFAULT);
            m_desc.ByteWidth = sizeof(UINT) * size;

            m_data.pSysMem = data;
            m_data.SysMemPitch = 0;
            m_data.SysMemSlicePitch = 0;

            m_elementCount = size;
        }

        VOID IndexBuffer::Bind(VOID) 
        {
            chimera::d3d::GetContext()->IASetIndexBuffer(m_pBuffer, DXGI_FORMAT_R32_UINT, 0);
        }

        IndexBuffer::~IndexBuffer() {}

        VertexBuffer::VertexBuffer(VOID) {}

        VertexBuffer::VertexBuffer(CONST VOID* data, UINT vertexCount, UINT stride, D3D11_CPU_ACCESS_FLAG cpuAccessFlags)
        {
            SetData(data, vertexCount, stride, cpuAccessFlags);
        }

        VOID VertexBuffer::SetData(CONST VOID* data, UINT vertexCount, UINT stride, D3D11_CPU_ACCESS_FLAG cpuAccessFlags /* = */ )
        {
            m_offset = 0;
            m_stride = stride; 
            m_elementCount = vertexCount;
            SetCPUAccess(cpuAccessFlags);
            if(cpuAccessFlags == D3D11_CPU_ACCESS_WRITE)
            {
                SetUsage(D3D11_USAGE_DYNAMIC);
            }
            SetBindflags(D3D11_BIND_VERTEX_BUFFER);

            m_desc.ByteWidth = vertexCount * stride;

            m_data.pSysMem = data;
            m_data.SysMemPitch = 0;
            m_data.SysMemSlicePitch = 0;
        }

        /*
        VertexBuffer::VertexBuffer(CONST VOID* data, UINT vertexCount, UINT stride) 
        {
            SetBindflags(D3D11_BIND_VERTEX_BUFFER);

            m_offset = 0;
            m_stride = stride;

            m_elementCount = vertexCount;

            this->m_desc.ByteWidth = vertexCount * stride;

            this->m_data.pSysMem = data;
            this->m_data.SysMemPitch = 0;
            this->m_data.SysMemSlicePitch = 0;
        } */

        VOID VertexBuffer::Bind(VOID)
        {
            chimera::d3d::GetContext()->IASetVertexBuffers(0, 1, &m_pBuffer, &m_stride, &m_offset);
        }

        VertexBuffer::~VertexBuffer() {}

        VertexBufferHandle::VertexBufferHandle(VOID) : m_pVertexBuffer(NULL)
        {
        }

        UINT VertexBufferHandle::VGetByteCount(VOID) CONST
        {
            return m_pVertexBuffer->GetElementCount() * m_pVertexBuffer->GetStride() * sizeof(FLOAT);
        }

        VOID VertexBufferHandle::SetVertexData(CONST VOID* data, UINT vertexCount, UINT stride, D3D11_CPU_ACCESS_FLAG cpuAccessFlags /* = */ )
        {
            if(!m_pVertexBuffer)
            {
                m_pVertexBuffer = new chimera::d3d::VertexBuffer();
            }
            m_pVertexBuffer->SetData(data, vertexCount, stride, cpuAccessFlags);
        }

        BOOL VertexBufferHandle::VCreate(VOID)
        {
            m_pVertexBuffer->Create();
            return TRUE;
        }

        VOID VertexBufferHandle::VDestroy(VOID)
        {
            SAFE_DELETE(m_pVertexBuffer);
        }

        chimera::d3d::VertexBuffer* VertexBufferHandle::GetBuffer(VOID) CONST
        {
            return m_pVertexBuffer;
        }

        Geometry::Geometry(BOOL ownsInstanceBuffer /* = false */) 
            : m_pVertexBuffer(NULL), m_pIndexBuffer(NULL), m_ownsInstanceBuffer(ownsInstanceBuffer), m_pInstanceBuffer(NULL), m_elementCount(0), m_initialized(FALSE)
        {
            m_pDrawer = DEFAULT_DRAWER;
        }

        VOID Geometry::VSetIndexBuffer(CONST UINT* indices, UINT size) 
        {
            m_pIndexBuffer = new chimera::d3d::IndexBuffer(indices, size);
        }

        VOID Geometry::VSetVertexBuffer(CONST FLOAT* vertices, UINT count, UINT stride, BOOL cpuWrite) 
        {
            m_pVertexBuffer = new chimera::d3d::VertexBuffer(vertices, count, stride, cpuWrite ? D3D11_CPU_ACCESS_WRITE : (D3D11_CPU_ACCESS_FLAG)0);
        }

        VOID Geometry::VSetTopology(GeometryTopology primType)
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

        UINT Geometry::VGetByteCount(VOID) CONST
        {
            UINT indexBytes = 0;
            UINT instancedBytes = 0;
            if(m_pIndexBuffer)
            {
                indexBytes = sizeof(UINT) * m_pIndexBuffer->GetElementCount();
            }
            if(m_pInstanceBuffer)
            {
                instancedBytes = m_pInstanceBuffer->GetElementCount() * m_pInstanceBuffer->GetStride() * sizeof(FLOAT);
            }
            return m_pVertexBuffer->GetElementCount() * m_pVertexBuffer->GetStride() * sizeof(FLOAT) + instancedBytes + indexBytes;
        }

        VertexBuffer* Geometry::GetVertexBuffer(VOID)
        {
            return m_pVertexBuffer;
        }

        IDeviceBuffer* Geometry::VGetVertexBuffer(VOID)
        {
            return m_pVertexBuffer;
        }

        BOOL Geometry::VCreate(VOID)
        {
           // if(VIsReady()) return TRUE; //TODO needed?

            m_pVertexBuffer->Create();
        
            m_elementCount = m_pVertexBuffer->GetElementCount();

            if(m_pIndexBuffer)
            {
                m_pDrawer = DEFAULT_DRAWER;
                m_pIndexBuffer->Create();
                m_elementCount = m_pIndexBuffer->GetElementCount();
            }
            else
            {
                m_pDrawer = ARRAY_DRAWER;
            }

            if(m_pInstanceBuffer)
            {
                m_pInstanceBuffer->Create();
                m_pDrawer = INSTANCED_DRAWER;
            }

            return TRUE;
        }

        VOID Geometry::VDraw(VOID)
        {
            m_pDrawer->VDraw(this, m_elementCount, 0, 0);
            /*if(m_pInstanceBuffer)
            {
                d3d::GetContext()->DrawIndexedInstanced(this->m_pIndexBuffer->GetElementCount(), this->m_pInstanceBuffer->GetElementCount(), 0, 0, 0);
            }
            else
            {
                d3d::GetContext()->DrawIndexed(this->m_pIndexBuffer->GetElementCount(), 0, 0);
            } */
        }

        VOID Geometry::VDraw(UINT start, UINT count)
        {
            m_pDrawer->VDraw(this, count, start, 0);
            /*if(m_pInstanceBuffer)
            {

                d3d::GetContext()->DrawIndexedInstanced(count, this->m_pInstanceBuffer->GetElementCount(), start, 0, 0);
            }
            else
            {
                d3d::GetContext()->DrawIndexed(count, start, 0);
            } */
        }

        VOID Geometry::VBind(VOID)
        {
            if(m_psCurrentBound != this)
            {
                if(m_pIndexBuffer)
                {
                    this->m_pIndexBuffer->Bind();
                }

                if(m_pInstanceBuffer)
                {
                    ID3D11Buffer* buffer[2];
                    buffer[0] = m_pVertexBuffer->GetBuffer();
                    buffer[1] = m_pInstanceBuffer->GetBuffer();

                    UINT strides[2];
                    strides[0] = m_pVertexBuffer->GetStride();
                    strides[1] = m_pInstanceBuffer->GetStride();

                    UINT offsets[2];
                    offsets[0] = m_pVertexBuffer->GetOffset();
                    offsets[1] = m_pInstanceBuffer->GetOffset();

                    chimera::d3d::GetContext()->IASetVertexBuffers(0, 2, buffer, strides, offsets);
                }
                else
                {
                    this->m_pVertexBuffer->Bind();
                }

                m_psCurrentBound = this;

                chimera::d3d::GetContext()->IASetPrimitiveTopology(m_primType);
            }
        }

        VOID Geometry::VDestroy(VOID)
        {
            SAFE_DELETE(this->m_pIndexBuffer);
            SAFE_DELETE(this->m_pVertexBuffer);
            if(m_ownsInstanceBuffer)
            {
                SAFE_DELETE(this->m_pInstanceBuffer);
            }
        }

        VertexBuffer* Geometry::GetInstanceBuffer(VOID)
        {
            return m_pInstanceBuffer;
        }

        IndexBuffer* Geometry::GetIndexBuffer(VOID)
        {
            return m_pIndexBuffer;
        }

        VOID Geometry::SetOwnsInstanceBuffer(BOOL owns)
        {
            m_ownsInstanceBuffer = owns;
        }

        VOID Geometry::VAddInstanceBuffer(FLOAT* data, UINT count, UINT stride)
        {
            if(!m_ownsInstanceBuffer)
            {
                LOG_CRITICAL_ERROR("Geometry can't own an instance buffer!");
            }
            m_pInstanceBuffer = new chimera::d3d::VertexBuffer(data, count, stride);
            m_pDrawer = INSTANCED_DRAWER;
        }

        VOID Geometry::SetInstanceBuffer(chimera::d3d::VertexBuffer* buffer)
        {
            this->m_pInstanceBuffer = buffer;
            if(m_psCurrentBound == this)
            {
                m_psCurrentBound = NULL;
            }

            m_pDrawer = buffer != NULL ? INSTANCED_DRAWER : DEFAULT_DRAWER;
        }

        VOID Geometry::DeleteRawData(VOID)
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

        Geometry::~Geometry(VOID) 
        {
            VDestroy();
        }
    }
}