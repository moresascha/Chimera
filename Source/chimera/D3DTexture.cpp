#include "D3DTexture.h"

namespace chimera
{
    namespace d3d
    {
        Texture::Texture(VOID) : m_pTextureView(NULL), m_pTexture(NULL)
        {
        }

        VOID* Texture::VGetDevicePtr(VOID) 
        {
            return (VOID*)m_pTexture;
        }

        VOID* Texture::VGetViewDevicePtr(VOID) 
        {
            return (VOID*)m_pTextureView;
        }


        VOID Texture::VDestroy(VOID)
        {
            SAFE_RELEASE(m_pTextureView);
            SAFE_RELEASE(m_pTexture);
        }

        Texture2D::Texture2D(VOID* data, UINT width, UINT height, DXGI_FORMAT format) 
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

        Texture2D::Texture2D(VOID) : m_rawData(NULL), m_byteSize(0) 
        {
            ZeroMemory(&m_texDesc, sizeof(D3D11_TEXTURE2D_DESC));
            m_texDesc.MipLevels = 1;
            m_texDesc.SampleDesc.Count = 1;
            m_texDesc.SampleDesc.Quality = 0;
            m_texDesc.ArraySize = 1;
        }

        BOOL Texture2D::VCreate(VOID) 
        {
//            if(VIsReady()) return TRUE;

            m_texDesc.BindFlags = GetBindflags();
            m_texDesc.CPUAccessFlags = GetCPUAccess();
            m_texDesc.Usage = GetUsage();
    
            if(m_texDesc.MiscFlags & D3D11_RESOURCE_MISC_GENERATE_MIPS)
            {
                m_texDesc.BindFlags |= D3D11_BIND_RENDER_TARGET;
                if(FAILED(chimera::d3d::GetDevice()->CreateTexture2D(&m_texDesc, NULL, &m_pTexture))) return FALSE;
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
                //if(FAILED(d3d::GetDevice()->CreateTexture2D(&m_texDesc, &subData, &m_pTexture))) return FALSE;
                //LOG_ERROR("asdasd");
                if(FAILED(chimera::d3d::GetDevice()->CreateTexture2D(&m_texDesc, pSubData, &m_pTexture))) return FALSE;
            }
    
            /*
            D3DX11_IMAGE_LOAD_INFO info;
            ZeroMemory(&info, sizeof(info));
            info.Format = m_texDesc.Format;
            info.Width = m_texDesc.Width;
            info.Height = m_texDesc.Height;
            info.BindFlags = m_texDesc.BindFlags;
            info.MipLevels = m_texDesc.MipLevels;
            info.MiscFlags = m_texDesc.MiscFlags;
            info.Usage = m_texDesc.Usage;
            if(FAILED(D3DX11CreateTextureFromMemory(
                d3d::GetDevice(), m_rawData, m_texDesc.Width * m_texDesc.Width * 4, &info, NULL, (ID3D11Resource**)&m_pTexture, NULL))) return FALSE;
            */
            if(GetBindflags() & D3D11_BIND_SHADER_RESOURCE)
            {
                D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
                ZeroMemory(&viewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
                viewDesc.Format = m_texDesc.Format;
                viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                viewDesc.Texture2D.MipLevels = -1;
                viewDesc.Texture2D.MostDetailedMip = 0;

                if(FAILED(chimera::d3d::GetDevice()->CreateShaderResourceView(m_pTexture, &viewDesc, &m_pTextureView))) return FALSE;

                if(m_texDesc.MiscFlags & D3D11_RESOURCE_MISC_GENERATE_MIPS)
                {
                    chimera::d3d::GetContext()->GenerateMips(m_pTextureView);
                }
            }
    
            m_rawData = NULL;
            return TRUE;
        }

        VOID Texture2D::SetHeight(UINT h)
        {
            m_texDesc.Height = h;
        }

        VOID Texture2D::SetWidth(UINT w)
        {
            m_texDesc.Width = w;
        }

        VOID Texture2D::SetFormat(DXGI_FORMAT format)
        {
            m_texDesc.Format = format;
        }

        VOID Texture2D::SetMipMapLevels(UINT levels)
        {
            m_texDesc.MipLevels = levels;
        }

        VOID Texture2D::SetMicsFlags(UINT flags)
        {
            m_texDesc.MiscFlags = flags;
        }

        VOID Texture2D::SetArraySize(UINT size)
        {
            m_texDesc.ArraySize = size;
        }

        VOID Texture2D::SetSamplerCount(UINT count)
        {
            m_texDesc.SampleDesc.Count = count;
        }

        VOID Texture2D::SetSamplerQuality(UINT quality)
        {
            m_texDesc.SampleDesc.Quality = quality;
        }

        Texture2D::~Texture2D() 
        {
        }
    }
}