#include "D3DTexture.h"

namespace chimera
{
    namespace d3d
    {
        Texture::Texture(void) : m_pTextureView(NULL), m_pTexture(NULL)
        {
        }

        void* Texture::VGetDevicePtr(void) 
        {
            return (void*)m_pTexture;
        }

        void* Texture::VGetViewDevicePtr(void) 
        {
            return (void*)m_pTextureView;
        }


        void Texture::VDestroy(void)
        {
            SAFE_RELEASE(m_pTextureView);
            SAFE_RELEASE(m_pTexture);
        }

        Texture2D::Texture2D(void* data, uint width, uint height, DXGI_FORMAT format) 
        {
            m_rawData = data;
            ZeroMemory(&m_texDesc, sizeof(D3D11_TEXTURE2D_DESC));
            SetBindflags(D3D11_BIND_SHADER_RESOURCE);
            m_texDesc.Format = format;
            m_texDesc.Width = width;
            m_texDesc.Height = height;
            m_texDesc.MipLevels = 1;
            m_texDesc.SampleDesc.Count = 1;
            m_texDesc.SampleDesc.Quality = 0;
            m_texDesc.ArraySize = 1;
        }

        Texture2D::Texture2D(void) : m_rawData(NULL), m_byteSize(0) 
        {
            ZeroMemory(&m_texDesc, sizeof(D3D11_TEXTURE2D_DESC));
            m_texDesc.MipLevels = 1;
            m_texDesc.SampleDesc.Count = 1;
            m_texDesc.SampleDesc.Quality = 0;
            m_texDesc.ArraySize = 1;
        }

        bool Texture2D::VCreate(void) 
        {
//            if(VIsReady()) return TRUE;

            m_texDesc.BindFlags = GetBindflags();
            m_texDesc.CPUAccessFlags = GetCPUAccess();
            m_texDesc.Usage = GetUsage();
    
            if(m_texDesc.MiscFlags & D3D11_RESOURCE_MISC_GENERATE_MIPS)
            {
                m_texDesc.BindFlags |= D3D11_BIND_RENDER_TARGET;
                if(FAILED(chimera::d3d::GetDevice()->CreateTexture2D(&m_texDesc, NULL, &m_pTexture))) return false;
                D3D11_BOX destBox;
                destBox.left = 0;
                destBox.right = m_texDesc.Width;
                destBox.top = 0;
                destBox.bottom = m_texDesc.Height;
                destBox.front = 0;
                destBox.back = 1;
                chimera::d3d::GetContext()->UpdateSubresource(m_pTexture, 0, &destBox, m_rawData, m_texDesc.Width * 4, 0);
            }
            else
            {
                D3D11_SUBRESOURCE_DATA* pSubData = NULL;
                D3D11_SUBRESOURCE_DATA subData;
                if(m_rawData)
                {
                    ZeroMemory(&subData, sizeof(D3D11_SUBRESOURCE_DATA));
                    subData.pSysMem = m_rawData;
                    subData.SysMemPitch = m_texDesc.Width * 4;
                    subData.SysMemSlicePitch = 0;
                    pSubData = &subData;

                }
                D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateTexture2D(&m_texDesc, pSubData, &m_pTexture));
            }

            if(GetBindflags() & D3D11_BIND_SHADER_RESOURCE)
            {
                D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
                ZeroMemory(&viewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
                viewDesc.Format = m_texDesc.Format;
                viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                viewDesc.Texture2D.MipLevels = -1;
                viewDesc.Texture2D.MostDetailedMip = 0;

                if(FAILED(chimera::d3d::GetDevice()->CreateShaderResourceView(m_pTexture, &viewDesc, &m_pTextureView))) return false;

                if(m_texDesc.MiscFlags & D3D11_RESOURCE_MISC_GENERATE_MIPS)
                {
                    chimera::d3d::GetContext()->GenerateMips(m_pTextureView);
                }
            }
    
            m_rawData = NULL;
            return true;
        }

        void Texture2D::SetHeight(uint h)
        {
            m_texDesc.Height = h;
        }

        void Texture2D::SetWidth(uint w)
        {
            m_texDesc.Width = w;
        }

        void Texture2D::SetFormat(DXGI_FORMAT format)
        {
            m_texDesc.Format = format;
        }

        void Texture2D::SetMipMapLevels(uint levels)
        {
            m_texDesc.MipLevels = levels;
        }

        void Texture2D::SetMicsFlags(uint flags)
        {
            m_texDesc.MiscFlags = flags;
        }

        void Texture2D::SetArraySize(uint size)
        {
            m_texDesc.ArraySize = size;
        }

        void Texture2D::SetSamplerCount(uint count)
        {
            m_texDesc.SampleDesc.Count = count;
        }

        void Texture2D::SetSamplerQuality(uint quality)
        {
            m_texDesc.SampleDesc.Quality = quality;
        }

        Texture2D::~Texture2D() 
        {
        }
    }
}