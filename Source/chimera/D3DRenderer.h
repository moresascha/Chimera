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
            float m_specularExpo;
            float m_illum;
            float m_textureSCale;
            float unused;
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
            int g_castShadow[4];
        };

        struct _LightingBuffer
        {
            XMFLOAT4X4 m_view;
            XMFLOAT4X4 m_iView;
            XMFLOAT4X4 m_projection[3]; //TODO
            XMFLOAT4 m_lightPos;
            XMFLOAT4 m_intensity;
            XMFLOAT4 m_ambient;
            XMFLOAT4 m_distances;
        };

        class AlbedoBuffer : public IAlbedoBuffer
        {

        private:
            RenderTarget* m_targets[Diff_SamplersCnt];

            ID3D11RenderTargetView* m_views[Diff_SamplersCnt];

            uint m_width, m_height;

            D3D11_VIEWPORT m_viewPort;

        public:
            AlbedoBuffer(uint w, uint h);

            void VClearAndBindRenderTargets(void);

            void VUnbindRenderTargets(void);

            void VOnRestore(uint w, uint h);

            IRenderTarget* VGetRenderTarget(Diff_RenderTarget stage);

            IRenderTarget* VGetDepthStencilTarget(void);

            ~AlbedoBuffer(void);
        };

        class Renderer : public IRenderer
        {
        private:
            ID3D11ShaderResourceView* m_currentSetSampler[MAX_SAMPLER];

            IRenderTarget* m_pDefaultRenderTarget;

            IConstShaderBuffer* m_constBuffer[BufferCnt];

            AlbedoBuffer* m_pDefShader;

            std::unique_ptr<IShaderCache> m_pShaderCache;

            float m_backColor[4];

            void Delete(void);

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

            IBlendState* m_pAlphaBlendState;

            IShaderProgram* m_screenQuadProgram;

            void SetSampler(TextureSlot startSlot, ID3D11ShaderResourceView** view, uint count, uint stage);

            void CreateDefaultShader(void);

        public:
            Renderer(void);

            bool VCreate(CM_WINDOW_CALLBACK cb, CM_INSTANCE instance, LPCWSTR wndTitle, uint width, uint height);

            void VDestroy(void);

            void VSetBackground(float r, float g, float b, float a) 
            {
                m_backColor[0] = r;
                m_backColor[1] = g;
                m_backColor[2] = b;
                m_backColor[3] = a;
            }

            CM_HWND VGetWindowHandle(void);

            uint VGetHeight(void);

            uint VGetWidth(void);

            void VResize(uint w, uint h);

            void VSetFullscreen(bool fullscreen);

            bool VOnRestore(void);

            void* VGetDevice(void);

            void* VGetContext(void);

            void VSetActorId(ActorId id);

            void VPreRender(void);

            void VPresent(void);

            void VPushAlphaBlendState(void);

            IShaderCache* VGetShaderCache(void) { return m_pShaderCache.get(); }

            void VPostRender(void);

            void VPushRasterState(IRasterState* state);

            void VPopRasterState(void);

            void VPushBlendState(IBlendState* state);

            void VPopBlendState(void);

            void VPushDepthStencilState(IDepthStencilState* rstate, uint stencilRef = 0);

            void VPopDepthStencilState(void);

            void VSetViewPort(uint w, uint h);

            IAlbedoBuffer* VGetAlbedoBuffer(void);

            void VSetViewTransform(const util::Mat4& mat, const util::Mat4& invMat, const util::Vec3& eyePos);

            void VSetProjectionTransform(const util::Mat4& mat, float distance);

            void VSetWorldTransform(const util::Mat4& mat);

            void VPushWorldTransform(const util::Mat4& mat);

            void VPopWorldTransform(void);

            void VPushViewTransform(const util::Mat4& mat, const util::Mat4& invMat, const util::Vec3& eyePos);

            void VPopViewTransform(void);

            void VPushProjectionTransform(const util::Mat4& mat, float distance);

            void VPopProjectionTransform(void);

            void VPushMaterial(IMaterial& mat);

            void VPopMaterial(void);

            void VSetDefaultMaterial(void);

            void VSetDefaultTexture(void);

            void VClearAndBindBackBuffer(void);

            void VBindBackBuffer(void);

            void VPushCurrentRenderTarget(IRenderTarget* rt);

            void VPopCurrentRenderTarget(void);

            IRenderTarget* VGetCurrentRenderTarget(void);

            void VSetDiffuseTexture(IDeviceTexture* texture);

            void VSetTexture(TextureSlot slot, IDeviceTexture* texture);

            void VSetTextures(TextureSlot startSlot, IDeviceTexture** texture, uint count);

            void SetPointLightShadowCubeMapSampler(ID3D11ShaderResourceView* view);

            void VSetNormalMapping(bool map);

            void SetCubeMapViews(const util::Mat4 mats[6]);

            void VSetLightSettings(const util::Vec4& color, const util::Vec3& position, float radius, bool castShadow);

            void VSetLightSettings(const util::Vec4& color, const util::Vec3& position, const util::Vec3& viewDir, float radius, float angel, float intensity, bool castShadow);

            void SetCSMSettings(const util::Mat4& view, const util::Mat4& iView, const util::Mat4 projection[3], const util::Vec3& lightPos, const float distances[3]);

            IConstShaderBuffer* VGetConstShaderBuffer(ConstShaderBufferSlot slot);

            void VDrawScreenQuad(int x, int y, int w, int h);

            void VDrawScreenQuad(void);

            void VDrawLine(int x, int y, int w, int h);

            ~Renderer(void);
        };
    }
}

