#include "stdafx.h"
#include "Geometry.h"
#include "Vec3.h"

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
            d3d::GetContext()->DrawIndexedInstanced(count, geo->GetInstanceBuffer()->GetElementCount(), start, vertexbase, 0);
        }
    };

    class DefaultDrawer : public GeometryDrawer
    {
    public:
        VOID VDraw(Geometry* geo, CONST UINT& count, CONST UINT& start, CONST UINT& vertexbase)
        {
            d3d::GetContext()->DrawIndexed(count, start, vertexbase);            
        }
    };

    class ArrayDrawer : public GeometryDrawer
    {
    public:
        VOID VDraw(Geometry* geo, CONST UINT& count, CONST UINT& start, CONST UINT& vertexbase)
        {
            d3d::GetContext()->Draw(count, start);
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
        ZeroMemory(&this->m_desc, sizeof(D3D11_BUFFER_DESC));
        ZeroMemory(&this->m_data, sizeof(D3D11_SUBRESOURCE_DATA));
    }

    VOID Buffer::Create(VOID) 
    {
        if(m_created) return;
        m_created = TRUE;
        m_desc.Usage = GetUsage();
        m_desc.CPUAccessFlags = GetCPUAccess();
        m_desc.BindFlags = GetBindflags();
        CHECK__(d3d::g_pDevice->CreateBuffer(&this->m_desc, &this->m_data, &this->m_pBuffer));
    }

    D3D11_MAPPED_SUBRESOURCE* Buffer::Map(VOID)
    {
        CHECK__(d3d::GetContext()->Map(m_pBuffer, 0, GetCPUAccess() == D3D11_CPU_ACCESS_WRITE ? D3D11_MAP_WRITE_DISCARD : D3D11_MAP_READ, 0, &m_mappedData));
        return &m_mappedData;
    }

    VOID Buffer::Unmap(VOID)
    {
        d3d::GetContext()->Unmap(m_pBuffer, 0);
    }

    VOID Buffer::DeleteRawData(VOID)
    {
        SAFE_ARRAY_DELETE(m_data.pSysMem);
    }

    Buffer::~Buffer(VOID) 
    {
        SAFE_RELEASE(this->m_pBuffer);
    }

    IndexBuffer::IndexBuffer(CONST UINT* data, UINT size) 
    {
        SetBindflags(D3D11_BIND_INDEX_BUFFER);
        SetUsage(D3D11_USAGE_DEFAULT);
        this->m_desc.ByteWidth = sizeof(UINT) * size;

        this->m_data.pSysMem = data;
        this->m_data.SysMemPitch = 0;
        this->m_data.SysMemSlicePitch = 0;

        this->m_elementCount = size;
    }

    VOID IndexBuffer::Bind(VOID) 
    {
        d3d::g_pContext->IASetIndexBuffer(this->m_pBuffer, DXGI_FORMAT_R32_UINT, 0);
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
        SetBindflags(D3D11_BIND_VERTEX_BUFFER);

        this->m_desc.ByteWidth = vertexCount * stride;

        this->m_data.pSysMem = data;
        this->m_data.SysMemPitch = 0;
        this->m_data.SysMemSlicePitch = 0;
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
        d3d::GetContext()->IASetVertexBuffers(0, 1, &this->m_pBuffer, &this->m_stride, &m_offset);
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
            m_pVertexBuffer = new d3d::VertexBuffer();
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

    d3d::VertexBuffer* VertexBufferHandle::GetBuffer(VOID) CONST
    {
        return m_pVertexBuffer;
    }

    Geometry::Geometry(BOOL ownsInstanceBuffer /* = false */) 
        : m_pVertexBuffer(NULL), m_pIndexBuffer(NULL), m_ownsInstanceBuffer(ownsInstanceBuffer), m_pInstanceBuffer(NULL), m_elementCount(0), m_initialized(FALSE)
    {
        m_pDrawer = DEFAULT_DRAWER;
    }

    VOID Geometry::SetIndexBuffer(CONST UINT* indices, UINT size, D3D_PRIMITIVE_TOPOLOGY primType) 
    {
        SetTopology(primType);
        this->m_pIndexBuffer = new d3d::IndexBuffer(indices, size);
    }

    VOID Geometry::SetVertexBuffer(CONST FLOAT* vertices, UINT count, UINT stride) 
    {
        this->m_pVertexBuffer = new d3d::VertexBuffer(vertices, count, stride);
    }

    VOID Geometry::SetTopology(D3D_PRIMITIVE_TOPOLOGY primType)
    {
        m_primType = primType;
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

    BOOL Geometry::VCreate(VOID)
    {
        if(IsReady()) return TRUE; //TODO needed?

        this->m_pVertexBuffer->Create();
        
        m_elementCount = m_pVertexBuffer->GetElementCount();

        if(m_pIndexBuffer)
        {
            m_pDrawer = DEFAULT_DRAWER;
            this->m_pIndexBuffer->Create();
            m_elementCount = m_pIndexBuffer->GetElementCount();
        }
        else
        {
            m_pDrawer = ARRAY_DRAWER;
        }

        if(m_pInstanceBuffer)
        {
            this->m_pInstanceBuffer->Create();
            m_pDrawer = INSTANCED_DRAWER;
        }

        return TRUE;
    }

    VOID Geometry::Draw(VOID)
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

    VOID Geometry::Draw(UINT start, UINT count)
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

    VOID Geometry::Bind(VOID)
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

                d3d::GetContext()->IASetVertexBuffers(0, 2, buffer, strides, offsets);
            }
            else
            {
                this->m_pVertexBuffer->Bind();
            }

            m_psCurrentBound = this;

            d3d::GetContext()->IASetPrimitiveTopology(this->m_primType);
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

    VOID Geometry::SetInstanceBuffer(FLOAT* data, UINT count, UINT stride)
    {
        if(!m_ownsInstanceBuffer)
        {
            LOG_CRITICAL_ERROR("Geometry can't own an instance buffer!");
        }
        this->m_pInstanceBuffer = new d3d::VertexBuffer(data, count, stride);
        m_pDrawer = INSTANCED_DRAWER;
    }

    VOID Geometry::SetInstanceBuffer(d3d::VertexBuffer* buffer)
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
