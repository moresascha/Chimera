#pragma once
#include "stdafx.h"
#include "D3DGraphics.h"
namespace chimera
{
    class Mat4;
    namespace d3d
    {
        class ConstBuffer : public IConstShaderBuffer
        {
        private:
            ID3D11Buffer* m_buffer;
            D3D11_MAPPED_SUBRESOURCE m_ressource;
            UINT m_byteSize;

        public:
            ConstBuffer(VOID);

            VOID VInit(UINT byteSize, VOID* data = 0);

            VOID* VMap(VOID);

            VOID VSetData(VOID* data);

            VOID VSetFromMatrix(CONST util::Mat4& mat);

            VOID VUnmap(VOID);

            VOID VActivate(ConstShaderBufferSlot slot, UINT shader = ACTIVATE_ALL);

            VOID* VGetDevicePtr(VOID);

            ID3D11Buffer* GetBuffer(VOID);

            ~ConstBuffer(VOID);
        };
    }
}