#pragma once
#include "stdafx.h"
#include "D3DGraphics.h"

namespace chimera
{
    namespace d3d
    {
        class RenderTarget : public D3DResource, public IRenderTarget
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

            IDeviceTexture* m_pDevTexture;
            IDeviceTexture* m_pDevDepthStencilTexture;

            VOID Delete(VOID);

        public:
            RenderTarget(BOOL depthOnly = FALSE);

            BOOL VOnRestore(UINT width, UINT height, GraphicsFormat format, BOOL depthBuffer = TRUE, BOOL cubeMap = FALSE, UINT arraySize = 1);

            VOID VClear(VOID);

            VOID VBind(VOID);

            VOID VSetClearColor(FLOAT r, FLOAT g, FLOAT b, FLOAT a);

            UINT VGetWidth(VOID) { return m_w; }

            UINT VGetHeight(VOID) { return m_h; }

            IDeviceTexture* VGetTexture(VOID);

            IDeviceTexture* VGetDepthStencilTexture(VOID);

            ID3D11ShaderResourceView* GetShaderRessourceView(VOID) CONST;

            ID3D11ShaderResourceView* GetShaderRessourceViewDepth(VOID) CONST;

            ID3D11RenderTargetView* GetRenderTargetView(VOID) CONST;

            ID3D11DepthStencilView* GetDepthStencilView(VOID) CONST;

            ID3D11Texture2D* GetTexture(VOID) CONST;

            ~RenderTarget(VOID);
        };
    }
}