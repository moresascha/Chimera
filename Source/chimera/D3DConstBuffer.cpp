#include "D3DConstBuffer.h"
#include "Mat4.h"
namespace chimera
{
    namespace d3d
    {
        ConstBuffer::ConstBuffer(VOID) : m_buffer(NULL) 
        {

        }

        VOID ConstBuffer::VInit(UINT byteSize, VOID* data) 
        {
            m_byteSize = byteSize;

            D3D11_BUFFER_DESC bDesc;
            bDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            bDesc.ByteWidth = byteSize;
            bDesc.Usage = D3D11_USAGE_DYNAMIC;
            bDesc.MiscFlags = 0;
            bDesc.StructureByteStride = 0;
            D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateBuffer(&bDesc, NULL, &m_buffer));

            if(data)
            {
                VSetData(data);
            }
        }

        VOID ConstBuffer::VSetData(VOID* data)
        {
            VOID* mapped = VMap();
            memcpy(mapped, data, m_byteSize);
            VUnmap();
        }

        VOID ConstBuffer::VSetFromMatrix(CONST util::Mat4& mat)
        {
            VSetData((VOID*)&mat.m_m);
        }

        VOID ConstBuffer::VActivate(ConstShaderBufferSlot slot, UINT shader)
        {
            chimera::d3d::GetContext()->VSSetConstantBuffers((UINT)slot, 1, &m_buffer);
            chimera::d3d::GetContext()->GSSetConstantBuffers((UINT)slot, 1, &m_buffer);
            chimera::d3d::GetContext()->PSSetConstantBuffers((UINT)slot, 1, &m_buffer);
            //for now only VS, PS and GS
        }

        VOID* ConstBuffer::VMap(VOID) 
        {
            chimera::d3d::GetContext()->Map(m_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &m_ressource);
            return m_ressource.pData;
        }

        VOID ConstBuffer::VUnmap(VOID)
        {
            chimera::d3d::GetContext()->Unmap(m_buffer, 0);
        }

        VOID* ConstBuffer::VGetDevicePtr(VOID) 
        {
            return (VOID*)m_buffer;
        }

        ID3D11Buffer* ConstBuffer::GetBuffer(VOID)
        {
            return m_buffer;
        }

        ConstBuffer::~ConstBuffer(VOID) 
        {
            SAFE_RELEASE(m_buffer);
        }
    }
}