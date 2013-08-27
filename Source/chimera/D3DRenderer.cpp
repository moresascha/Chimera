#include "D3DRenderer.h"
#include "GameApp.h"
#include "GameView.h"
#include "SceneGraph.h"

namespace d3d 
{

    D3DRenderer::D3DRenderer(VOID) : m_pDefaultRenderTarget(NULL)
    {
        for(UCHAR i = 0; i < BufferCnt; ++i)
        {
            m_constBuffer[i] = NULL;
        }
        for(UCHAR i = 0; i < MAX_SAMPLER; ++i)
        {
            m_currentSetSampler[i] = NULL;
        }

        m_pDefShader = new d3d::DeferredShader(d3d::g_width, d3d::g_height);

        /*m_defaultMaterial.m_ambient.Set(0.5f,0.5f,0.5f,1);
        m_defaultMaterial.m_diffuse.Set(1, 1, 1, 1);
        m_defaultMaterial.m_specular.Set(0, 0, 0, 0);
        m_defaultMaterial.m_texScale = 1;*/
        m_defaultMaterial.m_textureDiffuse = tbd::Resource("default.png");

        PushRasterizerState(d3d::g_pRasterizerStateFrontFaceSolid);
        PushBlendState(d3d::g_pBlendStateNoBlending);
    }

    HRESULT D3DRenderer::VOnRestore(VOID) 
    {
        Delete();
        m_pDefaultTexture = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_defaultMaterial.m_textureDiffuse));

        m_pDefShader->OnRestore(d3d::g_width, d3d::g_height);

        m_backColor[0] = 0; m_backColor[1] = 0; m_backColor[2] = 0; m_backColor[3] = 0;

        this->m_constBuffer[eModelBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eModelBuffer]->Init(sizeof(_ModelMatrixBuffer));

        this->m_constBuffer[eViewBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eViewBuffer]->Init(sizeof(_ViewMatrixBuffer));
        
        this->m_constBuffer[eProjectionBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eProjectionBuffer]->Init(sizeof(_ProjectionMatrixBuffer));

        this->m_constBuffer[eMaterialBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eMaterialBuffer]->Init(sizeof(_MaterialBuffer));

        this->m_constBuffer[ePointLightBuffer]  = new d3d::ConstBuffer();
        this->m_constBuffer[ePointLightBuffer]->Init(sizeof(_LightSettingsBuffer));

        this->m_constBuffer[eCubeMapViewsBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eCubeMapViewsBuffer]->Init(sizeof(_CubeMapViewsBuffer));

        this->m_constBuffer[eFontBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eFontBuffer]->Init(8 * sizeof(FLOAT));

        this->m_constBuffer[eBoundingGeoBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eBoundingGeoBuffer]->Init(4 * sizeof(FLOAT));

        this->m_constBuffer[eActorIdBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eActorIdBuffer]->Init(4 * sizeof(UINT));

        this->m_constBuffer[eSelectedActorIdBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eSelectedActorIdBuffer]->Init(4 * sizeof(UINT));

        this->m_constBuffer[eGuiColorBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eGuiColorBuffer]->Init(4 * sizeof(FLOAT));

        this->m_constBuffer[eHasNormalMapBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eHasNormalMapBuffer]->Init(4 * sizeof(UINT));

        this->m_constBuffer[eLightingBuffer] = new d3d::ConstBuffer();
        this->m_constBuffer[eLightingBuffer]->Init(sizeof(_LightingBuffer));

        ID3D11Buffer* buffer[BufferCnt];
        for(UCHAR i = 0; i < BufferCnt; ++i)
        {
            buffer[i] = m_constBuffer[i]->GetBuffer();
        }

        d3d::GetContext()->VSSetConstantBuffers(0, BufferCnt, buffer);
        d3d::GetContext()->PSSetConstantBuffers(0, BufferCnt, buffer);
        d3d::GetContext()->GSSetConstantBuffers(0, BufferCnt, buffer);

        return S_OK;
    }

    VOID D3DRenderer::SetDefaultTexture(VOID)
    {
        if(m_pDefaultTexture->IsReady())
        {
            m_pDefaultTexture->Update();
        }
        else
        {
            m_pDefaultTexture = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_defaultMaterial.m_textureDiffuse));
        }
        SetDiffuseSampler(m_pDefaultTexture->GetShaderResourceView());
        SetNormalMapping(FALSE);
    }

    VOID D3DRenderer::SetDefaultRasterizerState(ID3D11RasterizerState* state)
    {
        m_rasterStateStack.Clear();
        PushRasterizerState(state);
    }

    VOID D3DRenderer::SetDefaultMaterial(VOID)
    {
        VPushMaterial(m_defaultMaterial);
        SetDefaultTexture();
    }

    /*
    VOID D3DRenderer::TransferBuffer(std::shared_ptr<d3d::ShaderProgram> program, UINT destinationSlot, UINT sourceSlot) CONST 
    {
        ID3D11Buffer* buffer[4];
        buffer[VIEW_MATRIX_SLOT] = this->m_viewBuffer->GetBuffer();
        buffer[PROJECTION_MATRIX_SLOT] = this->m_projBuffer->GetBuffer();
        buffer[MODEL_MATRIX_SLOT] = this->m_modelBuffer->GetBuffer();
        buffer[MATERIAL_SLOT] = this->m_materialBuffer->GetBuffer();

        if(destinationSlot == -1)
        {
            program->SetConstBuffers(0, buffer, 4);
        }
        else
        {
            program->SetConstBuffers(destinationSlot, &buffer[sourceSlot], 1);
        }
    } */

    VOID D3DRenderer::VPreRender(VOID) 
    {
        SetSampler(eAmbientMaterialSpecGSampler, m_pDefShader->GetTarget(Diff_AmbientMaterialSpecGTarget)->GetShaderRessourceView(), 1, PS_Stage);
        SetSampler(eDiffuseColorSpecBSampler, m_pDefShader->GetTarget(Diff_DiffuseColorSpecBTarget)->GetShaderRessourceView(), 1, PS_Stage);
        SetSampler(eDiffuseMaterialSpecRSampler, m_pDefShader->GetTarget(Diff_DiffuseMaterialSpecRTarget)->GetShaderRessourceView(), 1, PS_Stage);
        SetSampler(eNormalsSampler, m_pDefShader->GetTarget(Diff_NormalsTarget)->GetShaderRessourceView(), 1, PS_Stage);
        SetSampler(eWorldPositionSampler, m_pDefShader->GetTarget(Diff_WorldPositionTarget)->GetShaderRessourceView(), 1, PS_Stage);
        SetSampler(eNormalColorSampler, m_pDefShader->GetTarget(Diff_ReflectionStrTarget)->GetShaderRessourceView(), 1, PS_Stage);
    }

    VOID D3DRenderer::VPostRender(VOID) 
    {
        SetSampler(eAmbientMaterialSpecGSampler, NULL, 1, PS_Stage);
        SetSampler(eDiffuseColorSpecBSampler, NULL, 1, PS_Stage);
        SetSampler(eDiffuseMaterialSpecRSampler, NULL, 1, PS_Stage);
        SetSampler(eNormalsSampler, NULL, 1, PS_Stage);
        SetSampler(eWorldPositionSampler, NULL, 1, PS_Stage);
        SetSampler(eNormalColorSampler, NULL, 1, PS_Stage);
    }

    VOID D3DRenderer::SetNormalMapping(BOOL map)
    {
        d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eHasNormalMapBuffer);
        FLOAT* data = (FLOAT*)buffer->Map();
        data[0] = map ? 1.0f : 0.0f;
        data[1] = map ? 1.0f : 0.0f;
        data[2] = map ? 1.0f : 0.0f;
        data[3] = map ? 1.0f : 0.0f;
        buffer->Unmap();
    }

    VOID D3DRenderer::SetCurrentRendertarget(d3d::RenderTarget* rt)
    {
        m_pDefaultRenderTarget = rt;
    }

    VOID D3DRenderer::ClearAndBindBackbuffer(VOID)
    {
        ClearBackbuffer();
        d3d::BindBackbuffer();
    }

    VOID D3DRenderer::ClearBackbuffer(VOID)
    {
        d3d::ClearBackBuffer(m_backColor);
    }

    VOID D3DRenderer::ActivateCurrentRendertarget(VOID)
    {
        m_pDefaultRenderTarget->Bind();
    }

    d3d::RenderTarget* D3DRenderer::GetCurrentrenderTarget(VOID)
    {
        return m_pDefaultRenderTarget;
    }

    VOID D3DRenderer::VSetWorldTransform(CONST util::Mat4& mat) 
    {
        this->m_constBuffer[eModelBuffer]->SetFromMatrix(mat);
    }

    VOID D3DRenderer::VPushWorldTransform(CONST util::Mat4& mat) 
    {
        this->m_constBuffer[eModelBuffer]->SetFromMatrix(mat);
    }

    VOID D3DRenderer::VPopWorldTransform(VOID)
    {

    }

    VOID D3DRenderer::VSetViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos)
    {
        m_viewMatrixStack.Clear();
        _ViewMatrixBuffer buffer;
        buffer.m_view = mat.m_m;
        buffer.m_invView = invMat.m_m;
        buffer.m_eyePos.x = eyePos.m_v.x;
        buffer.m_eyePos.y = eyePos.m_v.y;
        buffer.m_eyePos.z = eyePos.m_v.z;
        buffer.m_eyePos.w = 0;

        _ViewMatrixBuffer* vb = (_ViewMatrixBuffer*)this->m_constBuffer[eViewBuffer]->Map();
        *vb = buffer;
        this->m_constBuffer[eViewBuffer]->Unmap();
        //m_viewBuffer->SetFromMatrix(mat);
        m_viewMatrixStack.Push(buffer);
        app::g_pApp->GetHumanView()->GetSceneGraph()->ResetVisibility();
    }

    VOID D3DRenderer::VPushViewTransform(CONST util::Mat4& mat, CONST util::Mat4& invMat, CONST util::Vec3& eyePos)
    {
        //VSetViewTransform(mat, invMat, eyePos);

        _ViewMatrixBuffer buffer;
        buffer.m_view = mat.m_m;
        buffer.m_invView = invMat.m_m;
        buffer.m_eyePos.x = eyePos.m_v.x;
        buffer.m_eyePos.y = eyePos.m_v.y;
        buffer.m_eyePos.z = eyePos.m_v.z;
        buffer.m_eyePos.w = 0;

        m_viewMatrixStack.Push(buffer);
        _ViewMatrixBuffer* pBuffer = (_ViewMatrixBuffer*)m_constBuffer[eViewBuffer]->Map();
        *pBuffer = buffer;
        m_constBuffer[eViewBuffer]->Unmap();

        app::g_pApp->GetHumanView()->GetSceneGraph()->ResetVisibility();
    }

    VOID D3DRenderer::VPopViewTransform(VOID)
    {
        if(m_viewMatrixStack.Size() <= 1) 
        {
            return;
        }
        m_viewMatrixStack.Pop();
        _ViewMatrixBuffer buffer = m_viewMatrixStack.Peek();

        _ViewMatrixBuffer* pBuffer = (_ViewMatrixBuffer*)m_constBuffer[eViewBuffer]->Map();
        *pBuffer = buffer;
        m_constBuffer[eViewBuffer]->Unmap();

        app::g_pApp->GetHumanView()->GetSceneGraph()->ResetVisibility();
    }

    VOID D3DRenderer::VPushMaterial(tbd::IMaterial& mat) 
    {
        _MaterialBuffer* mb = (_MaterialBuffer*)this->m_constBuffer[eMaterialBuffer]->Map();
        mb->m_ambient = mat.VGetAmbient().m_v;
        mb->m_specular = mat.VGetSpecular().m_v;
        mb->m_diffuse = mat.VGetDiffuse().m_v;
        mb->m_specularExpo = mat.VGetSpecularExpo();
        mb->m_illum = mat.VGetReflectance();
        mb->m_textureSCale = mat.VGetTextureScale();
        this->m_constBuffer[eMaterialBuffer]->Unmap();
    }

    VOID D3DRenderer::VSetProjectionTransform(CONST util::Mat4& mat, FLOAT distance) 
    {
        m_projectionMatrixStack.Clear();
        _ProjectionMatrixBuffer buffer;
        buffer.m_projection = mat.m_m;
        buffer.m_viewDistance.x = buffer.m_viewDistance.y = buffer.m_viewDistance.z = buffer.m_viewDistance.w = distance;
        m_projectionMatrixStack.Push(buffer);
        _ProjectionMatrixBuffer* pBuffer = (_ProjectionMatrixBuffer*)m_constBuffer[eProjectionBuffer]->Map();
        *pBuffer = buffer;
        m_constBuffer[eProjectionBuffer]->Unmap();

        app::g_pApp->GetHumanView()->GetSceneGraph()->ResetVisibility();
    }

    VOID D3DRenderer::VPushProjectionTransform(CONST util::Mat4& mat, FLOAT distance) 
    {
        _ProjectionMatrixBuffer buffer;
        buffer.m_projection = mat.m_m;
        buffer.m_viewDistance.x = buffer.m_viewDistance.y = buffer.m_viewDistance.z = buffer.m_viewDistance.w = distance;
        m_projectionMatrixStack.Push(buffer);
        _ProjectionMatrixBuffer* pBuffer = (_ProjectionMatrixBuffer*)m_constBuffer[eProjectionBuffer]->Map();
        *pBuffer = buffer;
        m_constBuffer[eProjectionBuffer]->Unmap();

        app::g_pApp->GetHumanView()->GetSceneGraph()->ResetVisibility();
    }

    VOID D3DRenderer::VPopProjectionTransform(VOID)
    {
        if(m_projectionMatrixStack.Size() <= 1) 
        {
            return;
        }

        m_projectionMatrixStack.Pop();
        _ProjectionMatrixBuffer buffer = m_projectionMatrixStack.Peek();

        _ProjectionMatrixBuffer* pBuffer = (_ProjectionMatrixBuffer*)m_constBuffer[eProjectionBuffer]->Map();
        *pBuffer = buffer;
        m_constBuffer[eProjectionBuffer]->Unmap();

        app::g_pApp->GetHumanView()->GetSceneGraph()->ResetVisibility();
    }

    VOID D3DRenderer::VPushPrimitiveType(UINT type) 
    {
        LOG_CRITICAL_ERROR("remove this");
       // d3d::GetContext()->IASetPrimitiveTopology((D3D_PRIMITIVE_TOPOLOGY)type);
    }

    VOID D3DRenderer::SetActorId(UINT id)
    {
        d3d::ConstBuffer* buffer = GetBuffer(d3d::eActorIdBuffer);
        UINT* i = (UINT*)buffer->Map();
        i[0] = id;
        i[1] = i[2] = i[3] = 0;
        buffer->Unmap();
    }

    VOID D3DRenderer::SetDiffuseSampler(ID3D11ShaderResourceView* view, SamplerTargetStage stages)
    {
        SetSampler(eDiffuseColorSampler, view, 1, stages);
    }

    VOID D3DRenderer::SetPointLightShadowCubeMapSampler(ID3D11ShaderResourceView* view, SamplerTargetStage stages)
    {
        SetSampler(ePointLightShadowCubeMapSampler, view, 1, stages);
    }

    VOID D3DRenderer::SetSampler(SamplerSlot slot, ID3D11ShaderResourceView* view, UINT count, SamplerTargetStage stages)
    {
        if(m_currentSetSampler[slot] != view)
        {
            /*
            if(stages & VS_Stage)
            {
               d3d::GetContext()->VSSetShaderResources(slot, 1, &view);
            } */
            if(stages & PS_Stage)
            {
               d3d::GetContext()->PSSetShaderResources(slot, count, &view);
            }
            /*
            if(stages & GS_STAGE)
            {
                d3d::GetContext()->GSSetShaderResources(slot, 1, &view);
            } */
            m_currentSetSampler[slot] = view;
        }
    }

    VOID D3DRenderer::SetCubeMapViews(CONST util::Mat4 mats[6])
    {
        _CubeMapViewsBuffer* mb = (_CubeMapViewsBuffer*)this->m_constBuffer[eCubeMapViewsBuffer]->Map();
        for(UCHAR i = 0; i < 6; ++i)
        {
            mb->m_views[i] = mats[i].m_m;
        }
        this->m_constBuffer[eCubeMapViewsBuffer]->Unmap();
    }

    VOID D3DRenderer::SetLightSettings(CONST util::Vec4& color, CONST util::Vec3& position, FLOAT radius)
    {
        SetLightSettings(color, position, position, radius, 0.f, 1.f);
    }

    VOID D3DRenderer::SetLightSettings(CONST util::Vec4& color, CONST util::Vec3& position, CONST util::Vec3& viewDir, FLOAT radius, FLOAT angel, FLOAT intensity)
    {
        _LightSettingsBuffer* plb = (_LightSettingsBuffer*)this->m_constBuffer[ePointLightBuffer]->Map();
        plb->m_colorNRadiusW.x = color.x;
        plb->m_colorNRadiusW.y = color.y;
        plb->m_colorNRadiusW.z = color.z;
        plb->m_colorNRadiusW.w = radius;
        plb->m_position.x = position.x;
        plb->m_position.y = position.y;
        plb->m_position.z = position.z;
        plb->m_position.w = intensity;
        plb->m_viewDirNAngel.x = viewDir.x;
        plb->m_viewDirNAngel.y = viewDir.y;
        plb->m_viewDirNAngel.z = viewDir.z;
        plb->m_viewDirNAngel.w = angel;
        this->m_constBuffer[ePointLightBuffer]->Unmap();
    }

    VOID D3DRenderer::SetCSMSettings(CONST util::Mat4& view, CONST util::Mat4& iView, CONST util::Mat4 projections[3], CONST util::Vec3& lightPos, CONST FLOAT distances[3])
    {
        _LightingBuffer* lb = (_LightingBuffer*)m_constBuffer[eLightingBuffer]->Map();
        lb->m_view = view.m_m;
        lb->m_iView = iView.m_m;
        lb->m_projection[0] = projections[0].m_m;
        lb->m_projection[1] = projections[1].m_m;
        lb->m_projection[2] = projections[2].m_m;
        lb->m_lightPos.x = lightPos.x;
        lb->m_lightPos.y = lightPos.y;
        lb->m_lightPos.z = lightPos.z;
        lb->m_distances.x = distances[0];
        lb->m_distances.y = distances[1];
        lb->m_distances.z = distances[2];
        m_constBuffer[eLightingBuffer]->Unmap();
    }

    d3d::ConstBuffer* D3DRenderer::GetBuffer(ConstBufferSlot slot)
    {
        return m_constBuffer[slot];
    }

    VOID D3DRenderer::SetViewPort(UINT w, UINT h)
    {
        D3D11_VIEWPORT viewPort;
        viewPort.Height = (FLOAT)h;
        viewPort.Width = (FLOAT)w;
        viewPort.MinDepth = 0;
        viewPort.MaxDepth = 1;
        viewPort.TopLeftX = 0;
        viewPort.TopLeftY = 0;
        d3d::GetContext()->RSSetViewports(1, &viewPort);
    }

    VOID D3DRenderer::PushRasterizerState(ID3D11RasterizerState* state)
    {
        m_rasterStateStack.Push(state);
        d3d::GetContext()->RSSetState(state);
    }

    VOID D3DRenderer::PopRasterizerState(VOID)
    {
        if(m_rasterStateStack.Size() > 1)
        {
            m_rasterStateStack.Pop();
            d3d::GetContext()->RSSetState(m_rasterStateStack.Peek());
        }
    }

    VOID D3DRenderer::PushBlendState(ID3D11BlendState* state)
    {
        m_blendStateStack.Push(state);
        d3d::GetContext()->OMSetBlendState(state, NULL, -1);
    }

    VOID D3DRenderer::PopBlendState(VOID)
    {
        if(m_blendStateStack.Size() > 1)
        {
            m_blendStateStack.Pop();
            d3d::GetContext()->OMSetBlendState(m_blendStateStack.Peek(), NULL, -1);
        }
    }

    VOID D3DRenderer::Delete(VOID)
    {
        for(UCHAR i = 0; i < BufferCnt; ++i)
        {
            SAFE_DELETE(m_constBuffer[i]);
        }
    }

    D3DRenderer::~D3DRenderer(VOID) 
    {
        Delete();
        SAFE_DELETE(m_pDefShader);
    }

    //Defshader
    DeferredShader::DeferredShader(UINT w, UINT h)
    {
        for(UINT i = 0; i < Diff_SamplersCnt; ++i)
        {
            m_targets[i] = new d3d::RenderTarget();
        }
        OnRestore(w, h);
    }

    VOID DeferredShader::OnRestore(UINT w, UINT h)
    {
        this->m_height = h;
        this->m_width = w;
        /*for(UINT i = 0; i < Diff_SamplersCnt; ++i)
        {
            SAFE_DELETE(m_targets[i]);
            m_targets[i] = new d3d::RenderTarget;
        }*/
        m_targets[Diff_WorldPositionTarget]->OnRestore(this->m_width, this->m_height, DXGI_FORMAT_R32G32B32A32_FLOAT, TRUE);
        m_targets[Diff_NormalsTarget]->OnRestore(this->m_width, this->m_height, DXGI_FORMAT_R32G32B32A32_FLOAT, FALSE);
        m_targets[Diff_DiffuseMaterialSpecRTarget]->OnRestore(this->m_width, this->m_height, DXGI_FORMAT_R16G16B16A16_FLOAT, FALSE);
        m_targets[Diff_AmbientMaterialSpecGTarget]->OnRestore(this->m_width, this->m_height, DXGI_FORMAT_R16G16B16A16_FLOAT, FALSE);
        m_targets[Diff_DiffuseColorSpecBTarget]->OnRestore(this->m_width, this->m_height, DXGI_FORMAT_R16G16B16A16_FLOAT, FALSE);
        m_targets[Diff_ReflectionStrTarget]->OnRestore(this->m_width, this->m_height, DXGI_FORMAT_R16_FLOAT, FALSE);

        for(UINT i = 0; i < Diff_SamplersCnt; ++i)
        {
            m_views[i] = m_targets[i]->GetRenderTargetView();
        }

        m_viewPort.MinDepth = 0;
        m_viewPort.MaxDepth = 1;
        m_viewPort.TopLeftX = 0;
        m_viewPort.TopLeftY = 0;

        m_viewPort.Width = (FLOAT)this->m_width;
        m_viewPort.Height = (FLOAT)this->m_height;
    }

    ID3D11DepthStencilView* DeferredShader::GetDepthStencilView(VOID)
    {
        return m_targets[0]->GetDepthStencilView();
    }

    VOID DeferredShader::ClearAndBindRenderTargets(VOID)
    {
        for(UINT i = 0; i < Diff_SamplersCnt; ++i)
        {
            m_targets[i]->Clear();
        }

        d3d::GetContext()->RSSetViewports(1, &m_viewPort);

        d3d::GetContext()->OMSetRenderTargets(Diff_SamplersCnt, m_views, m_targets[0]->GetDepthStencilView());
    }

    /*
    VOID DeferredShader::DisableRenderTargets(VOID)
    {

    } */

    d3d::RenderTarget* DeferredShader::GetTarget(Diff_RenderTargets stage)
    {
        return m_targets[stage];
    }

    DeferredShader::~DeferredShader(VOID)
    {
        for(UINT i = 0; i < Diff_SamplersCnt; ++i)
        {
            SAFE_DELETE(m_targets[i]);
        }
    }
}