#include "RenderTarget.h"

namespace d3d {

RenderTarget::RenderTarget(BOOL depthOnly) : 
    m_pShaderRessourceView(NULL), 
    m_pTexture(NULL), 
    m_pDepthStencilTexture(NULL), 
    m_pRenderTargetView(NULL),
    m_pDepthStencilView(NULL),
    m_initialized(FALSE), 
    m_depthOnly(depthOnly),
    m_quality(0), 
    m_samples(1),
    m_pShaderRessourceViewDepth(NULL)
{
    SetBindflags(D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET);
    SetClearColor(0,0,0,0);
}

ID3D11ShaderResourceView* RenderTarget::GetShaderRessourceView(VOID) CONST
{
    return m_pShaderRessourceView;
}

ID3D11ShaderResourceView* RenderTarget::GetShaderRessourceViewDepth(VOID) CONST
{
    return m_pShaderRessourceViewDepth;
}

ID3D11RenderTargetView* RenderTarget::GetRenderTargetView(VOID) CONST
{
    return m_pRenderTargetView;
}

ID3D11DepthStencilView* RenderTarget::GetDepthStencilView(VOID) CONST
{
    return m_pDepthStencilView;
}

ID3D11Texture2D* RenderTarget::GetTexture(VOID) CONST
{
    return m_pTexture;
}

VOID RenderTarget::SetClearColor(FLOAT r, FLOAT g, FLOAT b, FLOAT a)
{
    m_clearColor[0] = r;
    m_clearColor[1] = g;
    m_clearColor[2] = b;
    m_clearColor[3] = a;
}

BOOL RenderTarget::OnRestore(UINT width, UINT height, enum DXGI_FORMAT format, BOOL depthBuffer, BOOL cubeMap, UINT arraySize) 
{
    if(m_initialized)
    {
        Delete();
    }

    m_initialized = TRUE;

    m_w = width;

    m_h = height;

    if(cubeMap)
    {
        m_samples = 1;
        m_quality = 0;
    }

    BOOL mipmap = (GetMiscflags() & D3D11_RESOURCE_MISC_GENERATE_MIPS) == D3D11_RESOURCE_MISC_GENERATE_MIPS;

    ZeroMemory(&m_textureDesc, sizeof(D3D11_TEXTURE2D_DESC));
    m_textureDesc.ArraySize = arraySize;
    m_textureDesc.BindFlags = GetBindflags();
    m_textureDesc.Format = format;
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
        CHECK__(d3d::GetDevice()->CreateTexture2D(&m_textureDesc, NULL, &m_pTexture));

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
            CHECK__(d3d::GetDevice()->CreateShaderResourceView(m_pTexture, &viewDesc, &m_pShaderRessourceView));
        }

        D3D11_RENDER_TARGET_VIEW_DESC rtvd;
        ZeroMemory(&rtvd, sizeof(D3D11_RENDER_TARGET_VIEW_DESC));
        rtvd.Format = m_textureDesc.Format;
        rtvd.ViewDimension = (m_textureDesc.ArraySize == 1 ? (m_samples > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D)
            : (m_samples > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY: D3D11_RTV_DIMENSION_TEXTURE2DARRAY));
        rtvd.Texture2DArray.FirstArraySlice = 0;
        rtvd.Texture2DArray.ArraySize = m_textureDesc.ArraySize;
        CHECK__(d3d::GetDevice()->CreateRenderTargetView(m_pTexture, &rtvd, &m_pRenderTargetView));
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
        CHECK__(d3d::GetDevice()->CreateTexture2D(&dsbDesc, NULL, &m_pDepthStencilTexture));

        D3D11_DEPTH_STENCIL_VIEW_DESC dsVDesc;
        ZeroMemory(&dsVDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));
        dsVDesc.Format = m_depthOnly ? DXGI_FORMAT_D32_FLOAT : dsbDesc.Format;
        dsVDesc.Texture2DArray.ArraySize = m_textureDesc.ArraySize;
        dsVDesc.ViewDimension = m_textureDesc.ArraySize == 1 ? (m_samples > 1 ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D) 
            : (m_samples > 1 ? D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY : D3D11_DSV_DIMENSION_TEXTURE2DARRAY);
        CHECK__(d3d::GetDevice()->CreateDepthStencilView(m_pDepthStencilTexture, &dsVDesc, &m_pDepthStencilView));
        
        if(m_depthOnly)
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
            ZeroMemory(&viewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
            viewDesc.Format = DXGI_FORMAT_R32_FLOAT;
            viewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            viewDesc.Texture2DArray.ArraySize = dsVDesc.Texture2DArray.ArraySize;
            viewDesc.Texture2D.MipLevels = dsbDesc.MipLevels;
            CHECK__(d3d::GetDevice()->CreateShaderResourceView(m_pDepthStencilTexture, &viewDesc, &m_pShaderRessourceViewDepth));
        }
    }

    return TRUE;
}

VOID RenderTarget::Clear(VOID) 
{
    if(m_pRenderTargetView)
    {
        d3d::GetContext()->ClearRenderTargetView(this->m_pRenderTargetView, m_clearColor);
    }

    if(this->m_pDepthStencilView)
    {
        d3d::GetContext()->ClearDepthStencilView(this->m_pDepthStencilView, D3D11_CLEAR_STENCIL | D3D11_CLEAR_DEPTH, 1, 1);
    }
}

VOID RenderTarget::Bind(VOID) 
{
    D3D11_VIEWPORT port;
    port.MinDepth = 0;
    port.MaxDepth = 1;
    port.TopLeftX = 0;
    port.TopLeftY = 0;
    port.Width = (FLOAT)this->m_textureDesc.Width;
    port.Height = (FLOAT)this->m_textureDesc.Height;
    d3d::GetContext()->RSSetViewports(1, &port);

    d3d::GetContext()->OMSetRenderTargets(1, &m_pRenderTargetView, m_pDepthStencilView);
}

VOID RenderTarget::Delete(VOID)
{
    SAFE_RELEASE(m_pDepthStencilView);
    SAFE_RELEASE(m_pRenderTargetView);
    SAFE_RELEASE(m_pShaderRessourceView);
    SAFE_RELEASE(m_pDepthStencilTexture);
    SAFE_RELEASE(m_pTexture);
    SAFE_RELEASE(m_pShaderRessourceViewDepth);
}
    
RenderTarget::~RenderTarget(VOID) 
{
    this->Delete();
}
};
