#include "PointLightNode.h"
#include "Vec3.h"
#include "Mat4.h"
#include "SceneGraph.h"
#include "ShaderProgram.h"
#include "GameApp.h"
#include "GeometryFactory.h"
#include "Components.h"
#include "D3DRenderer.h"
#include "GameLogic.h"

namespace chimera 
{
    chimera::RenderTarget* PointlightNode::g_pCubeMapRenderTarget = NULL;

    chimera::PointLightFrustum PointlightNode::g_frustum;

    PointlightNode::PointlightNode(ActorId actorid) : SceneNode(actorid)
    {
        //m_cubeMapRenderTarget = NULL;
        std::shared_ptr<chimera::Actor> actor = chimera::g_pApp->GetLogic()->VFindActor(actorid);
        this->m_lightComponent = m_actor->GetComponent<chimera::LightComponent>(chimera::LightComponent::COMPONENT_ID).lock();

        m_drawShadow = chimera::ShaderProgram::GetProgram("PointLightShadowMap").get();
        m_drawShadowInstanced = chimera::ShaderProgram::GetProgram("PointLightShadowMapInstanced").get();
        m_drawLighting = chimera::ShaderProgram::GetProgram("PointLight").get();

    }

    VOID PointlightNode::VOnActorMoved(VOID)
    {
        SceneNode::VOnActorMoved();
        XMMATRIX mat = XMMatrixPerspectiveFovLH(XM_PIDIV2, 1.0f, 0.01f, GetTransformation()->GetScale().x);
        XMStoreFloat4x4(&m_projection.m_m, mat);
    }

    VOID PointlightNode::VOnRestore(chimera::SceneGraph* graph)
    {
    
        util::Vec3& lightPos = util::Vec3();
        m_mats[0] = util::Mat4::createLookAtLH(lightPos, util::Vec3(1,0,0), util::Vec3(0,1,0)); //right

        m_mats[1] = util::Mat4::createLookAtLH(lightPos, util::Vec3(-1,0,0), util::Vec3(0,1,0)); //left

        m_mats[2] = util::Mat4::createLookAtLH(lightPos, util::Vec3(0,1,0), util::Vec3(0,0,-1)); //up

        m_mats[3] = util::Mat4::createLookAtLH(lightPos, util::Vec3(0,-1,0), util::Vec3(0,0,1)); //down

        m_mats[4] = util::Mat4::createLookAtLH(lightPos, util::Vec3(0,0,1), util::Vec3(0,1,0)); //front

        m_mats[5] = util::Mat4::createLookAtLH(lightPos, util::Vec3(0,0,-1), util::Vec3(0,1,0)); //back

        XMMATRIX mat = XMMatrixPerspectiveFovLH(XM_PIDIV2, 1.0f, 0.01f, GetTransformation()->GetScale().x);
        XMStoreFloat4x4(&m_projection.m_m, mat);
        //m_projection = util::Mat4::CreatePerspectiveLH(XM_PIDIV2, 1.0f, 0.01f, m_lightComponent->m_radius);
        /*
        SAFE_DELETE(m_cubeMapRenderTarget);

        m_cubeMapRenderTarget = new d3d::RenderTarget();

        UINT shadowMapSize = 2048; //TODO

        if(!m_cubeMapRenderTarget->Create(shadowMapSize, shadowMapSize, DXGI_FORMAT_R32_FLOAT, TRUE, TRUE, 6))
        {
            LOG_ERROR("Failed to create render target");
        } */
    }

    BOOL PointlightNode::VIsVisible(SceneGraph* graph)
    {
        return graph->GetFrustum()->IsInside(GetTransformation()->GetTranslation(),  GetTransformation()->GetScale().x);
    }

    UINT PointlightNode::VGetRenderPaths(VOID)
    {
        return eRenderPath_DrawLighting | eRenderPath_DrawPicking | eRenderPath_DrawBounding | eRenderPath_DrawEditMode | eRenderPath_DrawDebugInfo;
    }

    VOID PointlightNode::_VRender(chimera::SceneGraph* graph, RenderPath& path)
    {
        switch(path)
        {
        case eRenderPath_DrawLighting: 
            {
                if(!m_lightComponent->m_activated)
                {
                    return;
                }

                m_drawShadow->Bind();
                chimera::D3DRenderer* renderer = chimera::g_pApp->GetHumanView()->GetRenderer();
                renderer->VPushProjectionTransform(m_projection, GetTransformation()->GetScale().x);
                renderer->SetCubeMapViews(m_mats);
                renderer->SetLightSettings(m_lightComponent->m_color, GetTransformation()->GetTranslation(), util::Vec3(), GetTransformation()->GetScale().x, 0, m_lightComponent->m_intensity);

                renderer->SetPointLightShadowCubeMapSampler(NULL);
                g_pCubeMapRenderTarget->Bind();
                g_pCubeMapRenderTarget->Clear();
                //m_frustum.Transform(*m_transformation->GetTransformation());
                g_frustum.SetParams(GetTransformation()->GetScale().x, GetTransformation()->GetTranslation());
                graph->PushFrustum(&g_frustum);

                //d3d::GetContext()->RSSetState(d3d::g_pRasterizerStateBackFaceSolid);
                graph->OnRender(chimera::eRenderPath_DrawToShadowMap);

                m_drawShadowInstanced->Bind();

                graph->OnRender(chimera::eRenderPath_DrawToShadowMapInstanced);
                //d3d::GetContext()->RSSetState(d3d::g_pRasterizerStateFrontFaceSolid);

                graph->PopFrustum();
                renderer->VPopProjectionTransform();

                renderer->ActivateCurrentRendertarget();

                renderer->SetPointLightShadowCubeMapSampler(g_pCubeMapRenderTarget->GetShaderRessourceView());
            
                m_drawLighting->Bind();

                chimera::GetContext()->OMSetDepthStencilState(chimera::m_pNoDepthNoStencilState, 0);
                chimera::GetContext()->OMSetBlendState(chimera::g_pBlendStateBlendAdd, NULL, 0xffffff);
                GeometryFactory::GetGlobalScreenQuad()->Bind();
                GeometryFactory::GetGlobalScreenQuad()->Draw();
                chimera::GetContext()->OMSetBlendState(chimera::g_pBlendStateNoBlending, NULL, 0xffffff);
                chimera::GetContext()->OMSetDepthStencilState(chimera::m_pDepthNoStencilState, 0);

            } break;

        case eRenderPath_DrawBounding :
            {
                chimera::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                chimera::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                chimera::ConstBuffer* buffer = chimera::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(chimera::eBoundingGeoBuffer);
                XMFLOAT4* f  = (XMFLOAT4*)buffer->Map();
                f->x = 1;//m_transformation->GetTransformation()->GetScale().x;
                f->y = 0;
                f->z = 0;
                f->w = 0;
                buffer->Unmap();
                GeometryFactory::GetSphere()->Bind();
                GeometryFactory::GetSphere()->Draw();
            } break;

        case eRenderPath_DrawPicking : 
            {
                util::Mat4 m = *GetTransformation();
                m.SetScale(1);
                DrawPickingSphere(m_actor, &m, 1);
            } break;

        case eRenderPath_DrawEditMode :
            {
                util::Mat4 m = *GetTransformation();
                m.SetScale(1);
                DrawAnchorSphere(m_actor, &m, 1);
            } break;

        case eRenderPath_DrawDebugInfo : 
            {            
                std::stringstream ss;
                ss << "PointLight_";
                ss << m_actorId;
                DrawInfoTextOnScreen(graph->GetCamera().get(), GetTransformation(), ss.str());
                break;
            }
        }
    }

    BOOL PointlightNode::Create(VOID)
    {
        g_pCubeMapRenderTarget = new chimera::RenderTarget();

        UINT shadowMapSize = chimera::g_pApp->GetConfig()->GetInteger("iPointLightSMSize");

        if(!g_pCubeMapRenderTarget->OnRestore(shadowMapSize, shadowMapSize, DXGI_FORMAT_R32_FLOAT, TRUE, TRUE, 6))
        {
            LOG_CRITICAL_ERROR("Failed to create render target");
            return FALSE;
        }

        return TRUE;
    }

    VOID PointlightNode::Destroy(VOID)
    {
        SAFE_DELETE(g_pCubeMapRenderTarget);
    }

    PointlightNode::~PointlightNode(VOID)
    {
        //SAFE_DELETE(s_pCubeMapRenderTarget);
    }
};