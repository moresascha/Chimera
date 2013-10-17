#pragma once
#include "stdafx.h"
#include "tbdStack.h"
#include "D3DGraphics.h"
#include "D3DRenderTarget.h"

namespace chimera 
{
    namespace d3d
    {
#define MAX_SAMPLER 16

        struct _ViewMatrixBuffer 
        {
            XMFLOAT4X4 m_view;
            XMFLOAT4X4 m_invView;
            XMFLOAT4 m_eyePos;
        };

        struct _MaterialBuffer 
        {
            XMFLOAT4 m_ambient;
            XMFLOAT4 m_specular;
            XMFLOAT4 m_diffuse;
            FLOAT m_specularExpo;
            FLOAT m_illum;
            FLOAT m_textureSCale;
            FLOAT unused;
        };

        struct _ProjectionMatrixBuffer 
        {
            XMFLOAT4X4 m_projection;
            XMFLOAT4 m_viewDistance;
        };

        struct _ModelMatrixBuffer
        {
            XMFLOAT4X4 m_model;
        };

        struct _CubeMapViewsBuffer
        {
            XMFLOAT4X4 m_views[6];
        };

        struct _LightSettingsBuffer
        {
            XMFLOAT4 m_colorNRadiusW;
            XMFLOAT4 m_position;
            XMFLOAT4 m_viewDirNAngel;
        };

        struct _LightingBuffer
        {
            XMFLOAT4X4 m_view;
            XMFLOAT4X4 m_iView;
            XMFLOAT4X4 m_projection[3]; //TODO
            XMFLOAT4 m_lightPos;
            XMFLOAT4 m_distances;
        };

        class AlbedoBuffer : public IAlbedoBuffer
        {

        private:
            RenderTarget* m_targets[Diff_SamplersCnt];

            ID3D11RenderTargetView* m_views[Diff_SamplersCnt];

            UINT m_width, m_height;

            D3D11_VIEWPORT m_viewPort;

        public:
            AlbedoBuffer(UINT w, UINT h);

            VOID VClearAndBindRenderTargets(VOID);

            VOID VUnbindRenderTargets(VOID);

            VOID VOnRestore(UINT w, UINT h);

            IRenderTarget* VGetRenderTarget(Diff_RenderTarget stage);

            IRenderTarget* VGetDepthStencilTarget(VOID);

            ~AlbedoBuffer(VOID);
        };

        class Renderer : public IRenderer
        {
        private:
            ID3D11ShaderResourceView* m_currentSetSampler[MAX_SAMPLER];

            IRenderTarget* m_pDefaultRenderTarget;

            IConstShaderBuffer* m_constBuffer[BufferCnt];

            AlbedoBuffer* m_pDefShader;

            std::unique_ptr<IShaderCache> m_pShaderCache;

            FLOAT m_backColor[4];

            VOID Delete(VOID);

            IMaterial* m_pDefaultMaterial;

            std::shared_ptr<IDeviceTexture> m_pDefaultTexture;

            util::tbdStack<_ModelMatrixBuffer> m_modelMatrixStack;
            util::tbdStack<_ProjectionMatrixBuffer> m_projectionMatrixStack;
            util::tbdStack<_ViewMatrixBuffer> m_viewMatrixStack;
            util::tbdStack<IRasterState*> m_rasterStateStack;
            util::tbdStack<IBlendState*> m_blendStateStack;
            util::tbdStack<IDepthStencilState*> m_depthStencilStateStack;

            IRasterState* m_pDefaultRasterState;
            IBlendState* m_pDefaultBlendState;
            IDepthStencilState* m_pDefaultDepthStencilState;

            IShaderProgram* m_screenQuadProgram;

            VOID SetSampler(TextureSlot startSlot, ID3D11ShaderResourceView** view, UINT count, UINT stage);

            VOID CreateDefaultShader(VOID);

        public:
            Renderer(VOID);

            BOOL VCreate(CM_WINDOW_CALLBACK cb, CM_INSTANCE instance, LPCWSTR wndTitle, UINT width, UINT height);

            VOID VDestroy(VOID);

            VOID VSetBackground(FLOAT r, FLOAT g, FLOAT b, FLOAT a) 
            {
                m_backColor[0] = r;
                m_backColor[1] = g;
                m_backColor[2] = b;
                m_backColor[3] = a;
            }

            CM_HWND VGetWindowHandle(VOID);

            UINT VGetHeight(VOID);

            UINT VGetWidth(VOID);

            VOID VResize(UINT w, UINT h);

            BOOL VOnRestore(VOID);

            VOID VPreRender(VOID);

            VOID VPresent(VOID);

            IShaderCache* VGetShaderCache(VOID) { return m_pShaderCache.get(); }

            VOID VPostRender(VOID);

            VOID VPushRasterState(IRasterState* state);

            VOID VPopRasterState(VOID);

            VOID VPushBlendState(IBlendState* state);

            VOID VPopBlendState(VOID);

            VOID VPushDepthStencilState(IDepthStencilState* rstate, UINT stencilRef = 0);

            VOID VPopDepthStencilState(VOID);

            VOID VSetViewPort(UINT w, UINT h);

            IAlbedoBuffer* VGetAlbedoBuffer(VOID);

            VOID VSetViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos);

            VOID VSetProjectionTransform(CONST util::Mat4& mat, FLOAT distance);

            VOID VSetWorldTransform(CONST util::Mat4& mat);

            VOID VPushWorldTransform(CONST util::Mat4& mat);

            VOID VPopWorldTransform(VOID);

            VOID VPushViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos);

            VOID VPopViewTransform(VOID);

            VOID VPushProjectionTransform(CONST util::Mat4& mat, FLOAT distance);

            VOID VPopProjectionTransform(VOID);

            VOID VPushMaterial(IMaterial& mat);

            VOID VPopMaterial(VOID);

            VOID VSetDefaultMaterial(VOID);

            VOID VClearAndBindBackBuffer(VOID);

            VOID VBindBackBuffer(VOID);

            VOID VPushCurrentRenderTarget(IRenderTarget* rt);

            VOID VPopCurrentRenderTarget(VOID);

            IRenderTarget* VGetCurrentRenderTarget(VOID);

            VOID VSetDiffuseTexture(IDeviceTexture* texture);

            VOID VSetTexture(TextureSlot slot, IDeviceTexture* texture);

            VOID VSetTextures(TextureSlot startSlot, IDeviceTexture** texture, UINT count);

            VOID SetPointLightShadowCubeMapSampler(ID3D11ShaderResourceView* view);

            VOID VSetNormalMapping(BOOL map);

            VOID SetCubeMapViews(CONST util::Mat4 mats[6]);

            VOID SetLightSettings(CONST util::Vec4& color, CONST util::Vec3& position, FLOAT radius);

            VOID SetLightSettings(CONST util::Vec4& color, CONST util::Vec3& position, CONST util::Vec3& viewDir, FLOAT radius, FLOAT angel, FLOAT intensity);

            VOID SetCSMSettings(CONST util::Mat4& view, CONST util::Mat4& iView, CONST util::Mat4 projection[3], CONST util::Vec3& lightPos, CONST FLOAT distances[3]);

            IConstShaderBuffer* VGetConstShaderBuffer(ConstShaderBufferSlot slot);

            VOID VDrawScreenQuad(INT x, INT y, INT w, INT h);

            VOID VDrawScreenQuad(VOID);

            VOID VDrawLine(INT x, INT y, INT w, INT h);

            ~Renderer(VOID);
        };
    }
}

