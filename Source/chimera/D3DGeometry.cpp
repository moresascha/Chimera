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
                chimera::d3d::GetContext()->DrawIndexedInstanced(count, geo->VGetInstanceBuffer()->VGetElementCount(), start, vertexbase, 0);
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

        Buffer::Buffer(VOID) : m_pBuffer(0), m_created(FALSE), m_elementCount(0)
        {
            ZeroMemory(&m_desc, sizeof(D3D11_BUFFER_DESC));
            ZeroMemory(&m_data, sizeof(D3D11_SUBRESOURCE_DATA));
        }

        VOID Buffer::VCreate(VOID) 
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

        VOID Buffer::VSetData(CONST VOID* v, UINT bytes)
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

        IndexBuffer::IndexBuffer(VOID) { }

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

        VOID IndexBuffer::VBind(VOID) 
        {
            chimera::d3d::GetContext()->IASetIndexBuffer(m_pBuffer, DXGI_FORMAT_R32_UINT, 0);
        }

        IndexBuffer::~IndexBuffer() {}

        VertexBuffer::VertexBuffer(VOID) : m_stride(1), m_offset(0) { }

        VertexBuffer::VertexBuffer(UINT vertexCount, UINT stride, CONST VOID* data, BOOL cpuAccessFlags)
        {
            VInitParamater(vertexCount, stride, data, cpuAccessFlags);
        }

        VOID VertexBuffer::VInitParamater(UINT vertexCount, UINT stride, CONST VOID* data /* = NULL */, BOOL cpuAccessFlags /* = FALSE */)
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

        VOID VertexBuffer::VBind(VOID)
        {
            chimera::d3d::GetContext()->IASetVertexBuffers(0, 1, &m_pBuffer, &m_stride, &m_offset);
        }

        VertexBuffer::~VertexBuffer() {}

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
            m_pVertexBuffer = new chimera::d3d::VertexBuffer(count, stride, vertices, cpuWrite);
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
                indexBytes = sizeof(UINT) * m_pIndexBuffer->VGetElementCount();
            }
            if(m_pInstanceBuffer)
            {
                instancedBytes = m_pInstanceBuffer->VGetElementCount() * m_pInstanceBuffer->VGetStride() * sizeof(FLOAT);
            }
            return m_pVertexBuffer->VGetElementCount() * m_pVertexBuffer->VGetStride() * sizeof(FLOAT) + instancedBytes + indexBytes;
        }

        IVertexBuffer* Geometry::VGetVertexBuffer(VOID)
        {
            return m_pVertexBuffer;
        }

        BOOL Geometry::VCreate(VOID)
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

            if(m_pInstanceBuffer)
            {
                m_pInstanceBuffer->VCreate();
                m_pDrawer = INSTANCED_DRAWER;
            }

            return TRUE;
        }

        VOID Geometry::VDraw(VOID)
        {
            m_pDrawer->VDraw(this, m_elementCount, 0, 0);
        }

        VOID Geometry::VDraw(UINT start, UINT count)
        {
            m_pDrawer->VDraw(this, count, start, 0);
        }

        VOID Geometry::VBind(VOID)
        {
            if(m_psCurrentBound != this)
            {
                if(m_pIndexBuffer)
                {
                    m_pIndexBuffer->VBind();
                }

                if(m_pInstanceBuffer)
                {
                    ID3D11Buffer* buffer[2];
                    buffer[0] = (ID3D11Buffer*)m_pVertexBuffer->VGetDevicePtr();
                    buffer[1] = (ID3D11Buffer*)m_pInstanceBuffer->VGetDevicePtr();

                    UINT strides[2];
                    strides[0] = m_pVertexBuffer->VGetStride();
                    strides[1] = m_pInstanceBuffer->VGetStride();

                    UINT offsets[2];
                    offsets[0] = m_pVertexBuffer->VGetOffset();
                    offsets[1] = m_pInstanceBuffer->VGetOffset();

                    chimera::d3d::GetContext()->IASetVertexBuffers(0, 2, buffer, strides, offsets);
                }
                else
                {
                    m_pVertexBuffer->VBind();
                }

                m_psCurrentBound = this;
                
                chimera::d3d::GetContext()->IASetPrimitiveTopology(m_primType);
            }
        }

        VOID Geometry::VDestroy(VOID)
        {
            SAFE_DELETE(m_pIndexBuffer);
            SAFE_DELETE(m_pVertexBuffer);
            if(m_ownsInstanceBuffer)
            {
                SAFE_DELETE(m_pInstanceBuffer);
            }
        }

        IVertexBuffer* Geometry::VGetInstanceBuffer(VOID)
        {
            return m_pInstanceBuffer;
        }

        IDeviceBuffer* Geometry::VGetIndexBuffer(VOID)
        {
            return m_pIndexBuffer;
        }

        VOID Geometry::SetOwnsInstanceBuffer(BOOL owns)
        {
            m_ownsInstanceBuffer = owns;
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

        VOID Geometry::VSetInstanceBuffer(IVertexBuffer* buffer)
        {
            m_pInstanceBuffer = buffer;
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