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
            uint m_byteSize;

        public:
            ConstBuffer(void);

            void VInit(uint byteSize, void* data = 0);

            void* VMap(void);

            void VSetData(void* data);

            void VSetFromMatrix(const util::Mat4& mat);

            void VUnmap(void);

            void VActivate(ConstShaderBufferSlot slot, uint shader = ACTIVATE_ALL);

            void* VGetDevicePtr(void);

            ID3D11Buffer* GetBuffer(void);

            ~ConstBuffer(void);
        };
    }
}