#include "Picker.h"
#include "GameApp.h"
#include "D3DRenderer.h"
#include "Camera.h"
#include "GameView.h"
#include "SceneGraph.h"
namespace chimera
{
    ActorId ActorPicker::VPick(void) const
    {
        return m_currentActor;
    }

    void ActorPicker::VPostRender(void)
    {
        D3D11_MAPPED_SUBRESOURCE res;
        chimera::GetContext()->CopyResource(m_texture->GetTexture(), m_renderTarget->GetTexture());
        chimera::GetContext()->Flush();
        chimera::GetContext()->Map(m_texture->GetTexture(), 0, D3D11_MAP_READ, 0, &res);
        m_currentActor = ((uint*)(res.pData))[0];
        chimera::GetContext()->Unmap(m_texture->GetTexture(), 0);
        // app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(m_currentActor);
        chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eSelectedActorIdBuffer);
        uint* b = (uint*)buffer->Map();
        b[0] = m_currentActor;
        buffer->Unmap();
    }

    void ActorPicker::VRender(void)
    {
        m_shaderProgram->Bind();
        m_renderTarget->Bind();
        m_renderTarget->Clear();
        chimera::g_pApp->GetHumanView()->GetRenderer()->VPushProjectionTransform(m_projection, 1000.0f);
        chimera::g_pApp->GetHumanView()->GetSceneGraph()->OnRender(eRenderPath_DrawPicking);
        chimera::g_pApp->GetHumanView()->GetRenderer()->VPopProjectionTransform();
        /*d3d::BindBackbuffer();
        d3d::SetDefaultViewPort();
        LOG_CRITICAL_ERROR("todo"); */
    }

    bool ActorPicker::VHasPicked(void) const
    {
        return m_currentActor != CM_INVALID_ACTOR_ID;
    }

    bool ActorPicker::VCreate(void)
    {
        if(m_created)
        {
            return true;
        }

        m_created = true;

        m_shaderProgram = chimera::ShaderProgram::CreateProgram("Picking", L"Picking.hlsl", "Picking_VS", "Picking_PS", NULL).get();
        m_shaderProgram->SetInputAttr("POSITION", 0, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        m_shaderProgram->SetInputAttr("NORMAL", 1, 0, DXGI_FORMAT_R32G32B32_FLOAT);
        m_shaderProgram->SetInputAttr("TEXCOORD", 2, 0, DXGI_FORMAT_R32G32_FLOAT);
        m_shaderProgram->GenerateLayout();
        m_shaderProgram->Bind();

        m_renderTarget = new chimera::RenderTarget();
        m_renderTarget->SetUsage(D3D11_USAGE_DEFAULT);
        m_renderTarget->SetBindflags(D3D11_BIND_RENDER_TARGET);
        //m_renderTarget->SetClearColor(-1, -1, -1, -1);
        //m_renderTarget->SetCPUAccess(D3D11_CPU_ACCESS_READ);
        uint w = 1, h = 1;
        m_texture = new chimera::D3DTexture2D();
        m_texture->SetWidth(w);
        m_texture->SetHeight(h);
        m_texture->SetFormat(DXGI_FORMAT_R32_UINT);
        m_texture->SetUsage(D3D11_USAGE_STAGING);
        m_texture->SetCPUAccess(D3D11_CPU_ACCESS_READ);
        
        if(!m_texture->VCreate())
        {
            return false;
        }

        if(!m_renderTarget->OnRestore(w, h, DXGI_FORMAT_R32_UINT))
        {
            return false;
        }

        m_projection = util::Mat4::CreatePerspectiveLH(XM_PIDIV2, 1, 0.01f, 1000.0f);

        return true;
    }

    chimera::RenderTarget* ActorPicker::GetTarget(void)
    {
        return m_renderTarget;
    }

    ActorPicker::~ActorPicker(void)
    {
        SAFE_DELETE(m_renderTarget);
        SAFE_DELETE(m_texture);
    }
}
