#include "D3DRenderTarget.h"

namespace chimera
{
    namespace d3d
    {

        class RTDevTexture : public IDeviceTexture
        {
            void* m_pView;
            void* m_pTexture;
        public:
            RTDevTexture(void* view, void* texture) : m_pView(view), m_pTexture(texture)
            {

            }

            void* VGetDevicePtr()
            {
                return m_pTexture;
            }

            void* VGetViewDevicePtr(void)
            {
                return m_pView;
            }

            bool VCreate(void) { return true; }

            void VDestroy() {}
            
            uint VGetByteCount(void) const { return 0; }
        };

        RenderTarget::RenderTarget(bool depthOnly) : 
            m_pShaderRessourceView(NULL), 
            m_pTexture(NULL), 
            m_pDepthStencilTexture(NULL), 
            m_pRenderTargetView(NULL),
            m_pDepthStencilView(NULL),
            m_initialized(false), 
            m_depthOnly(depthOnly),
            m_quality(0), 
            m_samples(1),
            m_pShaderRessourceViewDepth(NULL),
            m_pDevTexture(NULL),
            m_pDevDepthStencilTexture(NULL)
        {
            SetBindflags(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET);
            VSetClearColor(0,0,0,0);
        }

        ID3D11ShaderResourceView* RenderTarget::GetShaderRessourceView(void) const
        {
            return m_pShaderRessourceView;
        }

        ID3D11ShaderResourceView* RenderTarget::GetShaderRessourceViewDepth(void) const
        {
            return m_pShaderRessourceViewDepth;
        }

        ID3D11RenderTargetView* RenderTarget::GetRenderTargetView(void) const
        {
            return m_pRenderTargetView;
        }

        ID3D11DepthStencilView* RenderTarget::GetDepthStencilView(void) const
        {
            return m_pDepthStencilView;
        }

        ID3D11Texture2D* RenderTarget::GetTexture(void) const
        {
            return m_pTexture;
        }

        void RenderTarget::VSetClearColor(float r, float g, float b, float a)
        {
            m_clearColor[0] = r;
            m_clearColor[1] = g;
            m_clearColor[2] = b;
            m_clearColor[3] = a;
        }

        bool RenderTarget::VOnRestore(uint width, uint height, GraphicsFormat format, bool depthBuffer, bool cubeMap, uint arraySize) 
        {
            if(m_initialized)
            {
                Delete();
            }

            m_initialized = true;

            m_w = width;

            m_h = height;

            if(cubeMap)
            {
                m_samples = 1;
                m_quality = 0;
            }

            bool mipmap = (GetMiscflags() & D3D11_RESOURCE_MISC_GENERATE_MIPS) == D3D11_RESOURCE_MISC_GENERATE_MIPS;

            ZeroMemory(&m_textureDesc, sizeof(D3D11_TEXTURE2D_DESC));
            m_textureDesc.ArraySize = arraySize;
            m_textureDesc.BindFlags = GetBindflags();
            m_textureDesc.Format = GetD3DFormatFromCMFormat(format);
            m_textureDesc.MipLevels = mipmap ? 0 : 1;
            m_textureDesc.SampleDesc.Count = m_samples;
            m_textureDesc.SampleDesc.Quality = m_quality;
            m_textureDesc.Width = width;
            m_textureDesc.Height = height;
            m_textureDesc.Usage = GetUsage();
            m_textureDesc.CPUAccessFlags = GetCPUAccess();
            m_textureDesc.MiscFlags = cubeMap ? D3D11_RESOURCE_MISC_TEXTURECUBE : GetMiscflags();

            if(!m_depthOnly)
            {
                D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateTexture2D(&m_textureDesc, NULL, &m_pTexture));

                if(GetBindflags() & D3D11_BIND_SHADER_RESOURCE)
                {
                    D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
                    ZeroMemory(&viewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
                    viewDesc.Format = m_textureDesc.Format;
                    viewDesc.ViewDimension = (cubeMap ? D3D11_SRV_DIMENSION_TEXTURECUBE : (m_textureDesc.ArraySize == 1 ? (m_samples > 1 ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D)
                        : (m_samples > 1 ? D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY: D3D11_SRV_DIMENSION_TEXTURE2DARRAY)));
                    viewDesc.Texture2D.MipLevels = mipmap ? -1 : m_textureDesc.MipLevels;
                    viewDesc.Texture2D.MostDetailedMip = 0;
                    viewDesc.Texture2DArray.ArraySize = m_textureDesc.ArraySize;
                    D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateShaderResourceView(m_pTexture, &viewDesc, &m_pShaderRessourceView));
                }

                D3D11_RENDER_TARGET_VIEW_DESC rtvd;
                ZeroMemory(&rtvd, sizeof(D3D11_RENDER_TARGET_VIEW_DESC));
                rtvd.Format = m_textureDesc.Format;
                rtvd.ViewDimension = (m_textureDesc.ArraySize == 1 ? (m_samples > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D)
                    : (m_samples > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY: D3D11_RTV_DIMENSION_TEXTURE2DARRAY));
                rtvd.Texture2DArray.FirstArraySlice = 0;
                rtvd.Texture2DArray.ArraySize = m_textureDesc.ArraySize;
                D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateRenderTargetView(m_pTexture, &rtvd, &m_pRenderTargetView));
            }

            if(depthBuffer)
            {
                D3D11_TEXTURE2D_DESC dsbDesc;
                ZeroMemory(&dsbDesc, sizeof(D3D11_TEXTURE2D_DESC));
                dsbDesc.Format = m_depthOnly ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_D24_UNORM_S8_UINT;
                dsbDesc.MipLevels = 1;//m_textureDesc.MipLevels;
                dsbDesc.Width = m_textureDesc.Width;
                dsbDesc.Height = m_textureDesc.Height;
                dsbDesc.SampleDesc.Quality = m_textureDesc.SampleDesc.Quality;
                dsbDesc.SampleDesc.Count = m_textureDesc.SampleDesc.Count;
                dsbDesc.Usage = D3D11_USAGE_DEFAULT;
                dsbDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | (m_depthOnly ? D3D11_BIND_SHADER_RESOURCE : 0);
                dsbDesc.ArraySize = m_textureDesc.ArraySize;
                dsbDesc.SampleDesc.Count = m_samples;
                dsbDesc.SampleDesc.Quality = m_quality;
                dsbDesc.MiscFlags = 0;//m_textureDesc.MiscFlags;
                D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateTexture2D(&dsbDesc, NULL, &m_pDepthStencilTexture));

                D3D11_DEPTH_STENCIL_VIEW_DESC dsVDesc;
                ZeroMemory(&dsVDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
                dsVDesc.Format = m_depthOnly ? DXGI_FORMAT_D32_FLOAT : dsbDesc.Format;
                dsVDesc.Texture2DArray.ArraySize = m_textureDesc.ArraySize;
                dsVDesc.ViewDimension = m_textureDesc.ArraySize == 1 ? (m_samples > 1 ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D) 
                    : (m_samples > 1 ? D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY : D3D11_DSV_DIMENSION_TEXTURE2DARRAY);
                D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateDepthStencilView(m_pDepthStencilTexture, &dsVDesc, &m_pDepthStencilView));

                if(m_depthOnly)
                {
                    D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
                    ZeroMemory(&viewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
                    viewDesc.Format = DXGI_FORMAT_R32_FLOAT;
                    viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                    viewDesc.Texture2DArray.ArraySize = dsVDesc.Texture2DArray.ArraySize;
                    viewDesc.Texture2D.MipLevels = dsbDesc.MipLevels;
                    D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateShaderResourceView(m_pDepthStencilTexture, &viewDesc, &m_pShaderRessourceViewDepth));
                }
            }

            m_pDevTexture = new RTDevTexture(m_pShaderRessourceView, m_pTexture);
            m_pDevDepthStencilTexture = new RTDevTexture(m_pDepthStencilView, m_pDevTexture);

            return true;
        }

        IDeviceTexture* RenderTarget::VGetTexture(void)
        {
            return m_pDevTexture;
        }

        IDeviceTexture* RenderTarget::VGetDepthStencilTexture(void)
        {
            return m_pDevDepthStencilTexture;
        }

        void RenderTarget::VClear(void) 
        {
            if(m_pRenderTargetView)
            {
                chimera::d3d::GetContext()->ClearRenderTargetView(m_pRenderTargetView, m_clearColor);
            }

            if(this->m_pDepthStencilView)
            {
                chimera::d3d::GetContext()->ClearDepthStencilView(m_pDepthStencilView, D3D11_CLEAR_STENCIL | D3D11_CLEAR_DEPTH, 1, 0);
            }
        }

        void RenderTarget::VBind(void) 
        {
            D3D11_VIEWPORT port;
            port.MinDepth = 0;
            port.MaxDepth = 1;
            port.TopLeftX = 0;
            port.TopLeftY = 0;
            port.Width = (float)m_textureDesc.Width;
            port.Height = (float)m_textureDesc.Height;
            chimera::d3d::GetContext()->RSSetViewports(1, &port);

            chimera::d3d::GetContext()->OMSetRenderTargets(1, &m_pRenderTargetView, m_pDepthStencilView);
        }

        void RenderTarget::Delete(void)
        {
            SAFE_DELETE(m_pDevDepthStencilTexture);
            SAFE_DELETE(m_pDevTexture);
            SAFE_RELEASE(m_pDepthStencilView);
            SAFE_RELEASE(m_pRenderTargetView);
            SAFE_RELEASE(m_pShaderRessourceView);
            SAFE_RELEASE(m_pDepthStencilTexture);
            SAFE_RELEASE(m_pTexture);
            SAFE_RELEASE(m_pShaderRessourceViewDepth);
        }

        RenderTarget::~RenderTarget(void) 
        {
            Delete();
        }
    }
}