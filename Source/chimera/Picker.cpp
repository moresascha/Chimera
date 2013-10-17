#include "Picker.h"
#include "GameApp.h"
#include "D3DRenderer.h"
#include "Camera.h"
#include "GameView.h"
#include "SceneGraph.h"
namespace chimera
{
    ActorId ActorPicker::VPick(VOID) CONST
    {
        return m_currentActor;
    }

    VOID ActorPicker::VPostRender(VOID)
    {
        D3D11_MAPPED_SUBRESOURCE res;
        chimera::GetContext()->CopyResource(m_texture->GetTexture(), m_renderTarget->GetTexture());
        chimera::GetContext()->Flush();
        chimera::GetContext()->Map(m_texture->GetTexture(), 0, D3D11_MAP_READ, 0, &res);
        m_currentActor = ((UINT*)(res.pData))[0];
        chimera::GetContext()->Unmap(m_texture->GetTexture(), 0);
        // app::g_pApp->GetHumanView()->GetRenderer()->SetActorId(m_currentActor);
        chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eSelectedActorIdBuffer);
        UINT* b = (UINT*)buffer->Map();
        b[0] = m_currentActor;
        buffer->Unmap();
    }

    VOID ActorPicker::VRender(VOID)
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

    BOOL ActorPicker::VHasPicked(VOID) CONST
    {
        return m_currentActor != CM_INVALID_ACTOR_ID;
    }

    BOOL ActorPicker::VCreate(VOID)
    {
        if(m_created)
        {
            return TRUE;
        }

        m_created = TRUE;

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
        UINT w = 1, h = 1;
        m_texture = new chimera::D3DTexture2D();
        m_texture->SetWidth(w);
        m_texture->SetHeight(h);
        m_texture->SetFormat(DXGI_FORMAT_R32_UINT);
        m_texture->SetUsage(D3D11_USAGE_STAGING);
        m_texture->SetCPUAccess(D3D11_CPU_ACCESS_READ);
        
        if(!m_texture->VCreate())
        {
            return FALSE;
        }

        if(!m_renderTarget->OnRestore(w, h, DXGI_FORMAT_R32_UINT))
        {
            return FALSE;
        }

        m_projection = util::Mat4::CreatePerspectiveLH(XM_PIDIV2, 1, 0.01f, 1000.0f);

        return TRUE;
    }

    chimera::RenderTarget* ActorPicker::GetTarget(VOID)
    {
        return m_renderTarget;
    }

    ActorPicker::~ActorPicker(VOID)
    {
        SAFE_DELETE(m_renderTarget);
        SAFE_DELETE(m_texture);
    }
}
