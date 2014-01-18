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

            bool m_initialized;
            bool m_depthOnly;

            uint m_samples;
            uint m_quality;

            float m_clearColor[4];

            uint m_w, m_h;

            IDeviceTexture* m_pDevTexture;
            IDeviceTexture* m_pDevDepthStencilTexture;

            void Delete(void);

        public:
            RenderTarget(bool depthOnly = false);

            bool VOnRestore(uint width, uint height, GraphicsFormat format, bool depthBuffer = true, bool cubeMap = false, uint arraySize = 1);

            void VClear(void);

            void VBind(void);

            void VSetClearColor(float r, float g, float b, float a);

            uint VGetWidth(void) { return m_w; }

            uint VGetHeight(void) { return m_h; }

            IDeviceTexture* VGetTexture(void);

            IDeviceTexture* VGetDepthStencilTexture(void);

            ID3D11ShaderResourceView* GetShaderRessourceView(void) const;

            ID3D11ShaderResourceView* GetShaderRessourceViewDepth(void) const;

            ID3D11RenderTargetView* GetRenderTargetView(void) const;

            ID3D11DepthStencilView* GetDepthStencilView(void) const;

            ID3D11Texture2D* GetTexture(void) const;

            ~RenderTarget(void);
        };
    }
}