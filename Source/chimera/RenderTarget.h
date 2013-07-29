#pragma once
#include "stdafx.h"
#include "d3d.h"
#include "D3DResource.h"

namespace d3d 
{

    class RenderTarget : public D3DResource
    {
    private:
        ID3D11Texture2D* m_pTexture;
        ID3D11Texture2D* m_pDepthStencilTexture;

        ID3D11ShaderResourceView* m_pShaderRessourceView;
        ID3D11ShaderResourceView* m_pShaderRessourceViewDepth;

        ID3D11RenderTargetView* m_pRenderTargetView;
        ID3D11DepthStencilView* m_pDepthStencilView;

        D3D11_TEXTURE2D_DESC m_textureDesc;

        BOOL m_initialized;
        BOOL m_depthOnly;

        UINT m_samples;
        UINT m_quality;

        FLOAT m_clearColor[4];

        UINT m_w, m_h;

        VOID Delete(VOID);

    public:
        RenderTarget(BOOL depthOnly = FALSE);

        BOOL OnRestore(UINT width, UINT height, enum DXGI_FORMAT format, BOOL depthBuffer = TRUE, BOOL cubeMap = FALSE, UINT arraySize = 1);

        VOID Clear(VOID);

        VOID Bind(VOID);

        VOID SetClearColor(FLOAT r, FLOAT g, FLOAT b, FLOAT a);

        UINT GetWidth(VOID) { return m_w; }

        UINT GetHeight(VOID) { return m_h; }

        ID3D11ShaderResourceView* GetShaderRessourceView(VOID) CONST;

        ID3D11ShaderResourceView* GetShaderRessourceViewDepth(VOID) CONST;

        ID3D11RenderTargetView* GetRenderTargetView(VOID) CONST;

        ID3D11DepthStencilView* GetDepthStencilView(VOID) CONST;

        ID3D11Texture2D* GetTexture(VOID) CONST;

        ~RenderTarget(VOID);
    };

};