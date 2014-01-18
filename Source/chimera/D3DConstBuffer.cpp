#include "D3DConstBuffer.h"
#include "Mat4.h"
namespace chimera
{
    namespace d3d
    {
        ConstBuffer::ConstBuffer(void) : m_buffer(NULL) 
        {

        }

        void ConstBuffer::VInit(uint byteSize, void* data) 
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

        void ConstBuffer::VSetData(void* data)
        {
            void* mapped = VMap();
            memcpy(mapped, data, m_byteSize);
            VUnmap();
        }

        void ConstBuffer::VSetFromMatrix(const util::Mat4& mat)
        {
            VSetData((void*)&mat.m_m);
        }

        void ConstBuffer::VActivate(ConstShaderBufferSlot slot, uint shader)
        {
            chimera::d3d::GetContext()->VSSetConstantBuffers((uint)slot, 1, &m_buffer);
            chimera::d3d::GetContext()->GSSetConstantBuffers((uint)slot, 1, &m_buffer);
            chimera::d3d::GetContext()->PSSetConstantBuffers((uint)slot, 1, &m_buffer);
            //for now only VS, PS and GS
        }

        void* ConstBuffer::VMap(void) 
        {
            chimera::d3d::GetContext()->Map(m_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &m_ressource);
            return m_ressource.pData;
        }

        void ConstBuffer::VUnmap(void)
        {
            chimera::d3d::GetContext()->Unmap(m_buffer, 0);
        }

        void* ConstBuffer::VGetDevicePtr(void) 
        {
            return (void*)m_buffer;
        }

        ID3D11Buffer* ConstBuffer::GetBuffer(void)
        {
            return m_buffer;
        }

        ConstBuffer::~ConstBuffer(void) 
        {
            SAFE_RELEASE(m_buffer);
        }
    }
}