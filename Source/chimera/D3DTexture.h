#pragma once
#include "stdafx.h"
#include "D3DGraphics.h"
namespace chimera
{
    namespace d3d
    {
        class Texture : public IDeviceTexture, public D3DResource
        {
        protected:
            ID3D11ShaderResourceView* m_pTextureView;
            ID3D11Texture2D* m_pTexture;
        public:
            Texture(void);
            void* VGetDevicePtr(void);
            void* VGetViewDevicePtr(void);
            virtual bool VCreate(void) = 0;
            void VDestroy();
            virtual uint VGetByteCount(void) const = 0;
            virtual ~Texture(void) {}
        };

        class Texture2D : public Texture
        {
        private:
            D3D11_TEXTURE2D_DESC m_texDesc;
            void* m_rawData;
            uint m_byteSize;

        public:
            //carefull using these, dont use these for images
            Texture2D(void* data, uint width, uint height, DXGI_FORMAT format);

            Texture2D(void);

            //VRamRessource Interface
            bool VCreate(void);

            uint VGetByteCount(void) const { return m_byteSize; }

            void SetData(void* data) { m_rawData = data; }

            void SetHeight(uint h);

            void SetWidth(uint w);

            void SetFormat(DXGI_FORMAT format);

            void SetMipMapLevels(uint levels);

            void SetMicsFlags(uint flags);

            void SetArraySize(uint size);

            void SetSamplerCount(uint count);

            void SetSamplerQuality(uint quality);

            const D3D11_TEXTURE2D_DESC& GetDescription(void) const { return m_texDesc; }

            ~Texture2D(void);
        };
    }
}