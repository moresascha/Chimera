#pragma once
#include "d3d.h"
#include "VRamManager.h"
#include "D3DResource.h"

namespace d3d 
{
    class Texture : public tbd::VRamHandle, public d3d::D3DResource
    {
    protected:
        ID3D11ShaderResourceView* m_pTextureView;
        ID3D11Texture2D* m_pTexture;
    public:
        Texture(VOID);
        ID3D11ShaderResourceView* GetShaderResourceView(VOID);
        ID3D11Texture2D* GetTexture(VOID);
        virtual BOOL VCreate(VOID) = 0;
        VOID VDestroy();
        virtual UINT VGetByteCount(VOID) CONST = 0;
        virtual ~Texture(VOID) {}
    };

    class Texture2D : public Texture
    {
    private:
        D3D11_TEXTURE2D_DESC m_texDesc;
        VOID* m_rawData;
        UINT m_byteSize;
    
    public:
        //carefull using these, dont use these for images
        //Texture2D(CONST WCHAR* file);
        Texture2D(VOID* data, UINT width, UINT height, DXGI_FORMAT format);
        Texture2D(VOID);

        //VRamRessource Interface
        BOOL VCreate(VOID);

        UINT VGetByteCount(VOID) CONST { return m_byteSize; }

        VOID SetData(VOID* data) { m_rawData = data; }

        VOID SetHeight(UINT h);

        VOID SetWidth(UINT w);

        VOID SetFormat(DXGI_FORMAT format);

        VOID SetMipMapLevels(UINT levels);

        VOID SetMicsFlags(UINT flags);

        VOID SetArraySize(UINT size);

        VOID SetSamplerCount(UINT count);

        VOID SetSamplerQuality(UINT quality);

        CONST D3D11_TEXTURE2D_DESC& GetDescription(VOID) CONST { return m_texDesc; }

        ~Texture2D(VOID);
    };
}



