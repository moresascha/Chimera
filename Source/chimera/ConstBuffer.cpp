#include "stdafx.h"
#include "ConstBuffer.h"

namespace d3d {
ConstBuffer::ConstBuffer(VOID) : m_buffer(0) {}

VOID ConstBuffer::Init(UINT byteSize, VOID* sharedData) {

    this->m_sharedData = sharedData;
    this->m_byteSize = byteSize;

    D3D11_BUFFER_DESC bDesc;
    bDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bDesc.ByteWidth = byteSize;
    bDesc.Usage = D3D11_USAGE_DYNAMIC;
    bDesc.MiscFlags = 0;
    bDesc.StructureByteStride = 0;
    CHECK__(d3d::GetDevice()->CreateBuffer(&bDesc, NULL, &this->m_buffer));
    if(sharedData)
    {
        Update();
    }
}

VOID ConstBuffer::Update(VOID) {
    if(!m_sharedData) 
    {
        LOG_INFO("No shared data");    
        return;
    }

    VOID* mapped = this->Map();
    memcpy(mapped, this->m_sharedData, this->m_byteSize);
    this->Unmap();
}

VOID ConstBuffer::SetFromMatrix(CONST util::Mat4& value) {
    XMFLOAT4X4* m_pBuffer = (XMFLOAT4X4*)this->Map();
    *m_pBuffer = value.m_m;
    this->Unmap();
}

VOID ConstBuffer::SetFromFloat4x4(CONST XMFLOAT4X4& value) {
    XMFLOAT4X4* m_pBuffer = (XMFLOAT4X4*)this->Map();
    *m_pBuffer = value;
    this->Unmap();
}

VOID ConstBuffer::Activate(UINT slot, UINT shader) {
    d3d::GetContext()->VSSetConstantBuffers(slot, 1, &this->m_buffer);
    d3d::GetContext()->GSSetConstantBuffers(slot, 1, &this->m_buffer);
    d3d::GetContext()->PSSetConstantBuffers(slot, 1, &this->m_buffer);
    //for now only VS, PS and GS
}

VOID* ConstBuffer::Map(VOID) {
     d3d::GetContext()->Map(this->m_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &this->m_ressource);
     return this->m_ressource.pData;
}

VOID ConstBuffer::Unmap(VOID) {
     d3d::GetContext()->Unmap(this->m_buffer, 0);
}

ID3D11Buffer* ConstBuffer::GetBuffer(VOID) {
     return this->m_buffer;
}

ConstBuffer::~ConstBuffer(VOID) {
    if(this->m_buffer) 
    {
        this->m_buffer->Release();
    }
}
}