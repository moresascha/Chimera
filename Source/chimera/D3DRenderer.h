#pragma once
#include "stdafx.h"
#include "Mat4.h"
#include "Vec4.h"
#include "ConstBuffer.h"
#include "Renderer.h"
#include "ShaderProgram.h"
#include "Texture.h"
#include "RenderTarget.h"
#include "tbdStack.h"

namespace d3d 
{
    #define MAX_SAMPLER 16
    enum SamplerTargetStage
    {
        VS_Stage = 1,
        PS_Stage = 2,
        GS_STAGE = 4,
    };

    enum ConstBufferSlot 
    {
        eViewBuffer,
        eProjectionBuffer,
        eModelBuffer,
        eMaterialBuffer,
        eCubeMapViewsBuffer,
        ePointLightBuffer,
        eFontBuffer,
        eBoundingGeoBuffer,
        eActorIdBuffer,
        eSelectedActorIdBuffer,
        eGuiColorBuffer,
        eHasNormalMapBuffer,
        eLightingBuffer,
        BufferCnt
    };

    enum SamplerSlot
    {
        eDiffuseColorSampler,
        eWorldPositionSampler,
        eNormalsSampler,
        eDiffuseMaterialSpecRSampler,
        eAmbientMaterialSpecGSampler,
        eDiffuseColorSpecBSampler,
        ePointLightShadowCubeMapSampler,
        eGuiSampler,
        eNormalColorSampler,
        eSceneSampler,
        eEffect0,
        eEffect1,
        eEffect2,
        eEffect3,
        SamplerCnt
    };

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
    };

    struct _LightingBuffer
    {
        XMFLOAT4X4 m_view;
        XMFLOAT4X4 m_iView;
        XMFLOAT4X4 m_projection[3]; //TODO
        XMFLOAT4 m_lightPos;
        XMFLOAT4 m_distances;
    };

    struct D3DState
    {
        //TODO Statestack
    };


    enum Diff_RenderTargets
    {
        Diff_WorldPositionTarget,
        Diff_NormalsTarget,
        Diff_DiffuseMaterialSpecRTarget,
        Diff_AmbientMaterialSpecGTarget,
        Diff_DiffuseColorSpecBTarget,
        Diff_ReflectionStrTarget,

        Diff_SamplersCnt
    };

    class DeferredShader
    {

    private:
        d3d::RenderTarget* m_targets[Diff_SamplersCnt];

        ID3D11RenderTargetView* m_views[Diff_SamplersCnt];

        UINT m_width, m_height;

        D3D11_VIEWPORT m_viewPort;

    public:
        DeferredShader(UINT w, UINT h);

        VOID ClearAndBindRenderTargets(VOID);

        VOID DisableRenderTargets(VOID);

        VOID OnRestore(UINT w, UINT h);

        d3d::RenderTarget* GetTarget(Diff_RenderTargets stage);

        ID3D11DepthStencilView* GetDepthStencilView(VOID);

        ~DeferredShader(VOID);
    };

    class D3DRenderer : public tbd::IRenderer
    {
    private:
        ID3D11ShaderResourceView* m_currentSetSampler[MAX_SAMPLER];

        d3d::RenderTarget* m_pDefaultRenderTarget;

        d3d::ConstBuffer* m_constBuffer[BufferCnt];

        d3d::DeferredShader* m_pDefShader;

        FLOAT m_backColor[4];

        VOID Delete(VOID);

        tbd::Material m_defaultMaterial;

        std::shared_ptr<d3d::Texture2D> m_pDefaultTexture;

        util::tbdStack<_ModelMatrixBuffer> m_modelMatrixStack;
        util::tbdStack<_ProjectionMatrixBuffer> m_projectionMatrixStack;
        util::tbdStack<_ViewMatrixBuffer> m_viewMatrixStack;
        util::tbdStack<ID3D11RasterizerState*> m_rasterStateStack;
        util::tbdStack<ID3D11BlendState*> m_blendStateStack;

    public:
        D3DRenderer(VOID);

        VOID VSetBackground(CHAR r, CHAR g, CHAR b, CHAR a) {
            m_backgroundColor.Set(r, g, b, a);
        }

        UINT VGetHeight(VOID) { return d3d::g_height; }

        UINT VGetWidth(VOID) { return d3d::g_width; }

        HRESULT VOnRestore(VOID);

        VOID VPreRender(VOID);

        VOID VPresent(VOID) { d3d::g_pSwapChain->Present(0, 0); }

        VOID VPostRender(VOID);

        VOID PushRasterizerState(ID3D11RasterizerState* state);

        VOID PopRasterizerState(VOID);

        VOID PushBlendState(ID3D11BlendState* state);

        VOID PopBlendState(VOID);

        VOID SetViewPort(UINT w, UINT h);

        VOID SetDefaultRasterizerState(ID3D11RasterizerState* state);

        d3d::DeferredShader* GetDeferredShader(VOID) CONST { return m_pDefShader; }

        VOID VSetViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos);

        VOID VSetProjectionTransform(CONST util::Mat4& mat, FLOAT distance);

        VOID VSetWorldTransform(CONST util::Mat4& mat);

        VOID VPushWorldTransform(CONST util::Mat4& mat);

        VOID VPopWorldTransform(VOID);

        VOID VPushViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos);

        VOID VPopViewTransform(VOID);

        VOID VPushProjectionTransform(CONST util::Mat4& mat, FLOAT distance);

        VOID VPopProjectionTransform(VOID);

        VOID VPushMaterial(tbd::IMaterial& mat);

        VOID SetDefaultMaterial(VOID);

        VOID ClearAndBindBackbuffer(VOID);

        VOID ClearBackbuffer(VOID);

        VOID SetDefaultTexture(VOID);

        VOID SetCurrentRendertarget(d3d::RenderTarget* rt);

        VOID ActivateCurrentRendertarget(VOID);

        d3d::RenderTarget* GetCurrentrenderTarget(VOID);

        VOID VPushPrimitiveType(UINT type);

        VOID SetActorId(UINT id);

        VOID SetNormalMapping(BOOL map);

        VOID SetDiffuseSampler(ID3D11ShaderResourceView* view, SamplerTargetStage stages = PS_Stage);

        VOID SetPointLightShadowCubeMapSampler(ID3D11ShaderResourceView* view, SamplerTargetStage stages = PS_Stage);

        VOID SetSampler(SamplerSlot slot, ID3D11ShaderResourceView* view, UINT count = 1,SamplerTargetStage stages = PS_Stage);

        VOID SetCubeMapViews(CONST util::Mat4 mats[6]);

        VOID SetLightSettings(CONST util::Vec4& color, CONST util::Vec3& position, FLOAT radius);

        VOID SetCSMSettings(CONST util::Mat4& view, CONST util::Mat4& iView, CONST util::Mat4 projection[3], CONST util::Vec3& lightPos, CONST FLOAT distances[3]);

        d3d::ConstBuffer* GetBuffer(ConstBufferSlot slot);

        ~D3DRenderer(VOID);
    };
}

