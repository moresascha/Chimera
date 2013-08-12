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

namespace tbd 
{
    d3d::RenderTarget* PointlightNode::g_pCubeMapRenderTarget = NULL;

    tbd::PointLightFrustum PointlightNode::g_frustum;

    PointlightNode::PointlightNode(ActorId actorid) : SceneNode(actorid)
    {
        //m_cubeMapRenderTarget = NULL;
        std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VFindActor(actorid);
        this->m_lightComponent = m_actor->GetComponent<tbd::LightComponent>(tbd::LightComponent::COMPONENT_ID).lock();

        m_drawShadow = d3d::ShaderProgram::GetProgram("PointLightShadowMap").get();
        m_drawShadowInstanced = d3d::ShaderProgram::GetProgram("PointLightShadowMapInstanced").get();
        m_drawLighting = d3d::ShaderProgram::GetProgram("PointLight").get();

    }

    VOID PointlightNode::VOnActorMoved(VOID)
    {
        SceneNode::VOnActorMoved();
        XMMATRIX mat = XMMatrixPerspectiveFovLH(XM_PIDIV2, 1.0f, 0.01f, GetTransformation()->GetScale().x);
        XMStoreFloat4x4(&m_projection.m_m, mat);
    }

    VOID PointlightNode::VOnRestore(tbd::SceneGraph* graph)
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
        return eDRAW_LIGHTING | eDRAW_PICKING | eDRAW_BOUNDING_DEBUG | eDRAW_EDIT_MODE | eDRAW_DEBUG_INFOS;
    }

    VOID PointlightNode::_VRender(tbd::SceneGraph* graph, RenderPath& path)
    {
        switch(path)
        {
        case eDRAW_LIGHTING: 
            {
                if(!m_lightComponent->m_activated)
                {
                    return;
                }

                m_drawShadow->Bind();
                d3d::D3DRenderer* renderer = app::g_pApp->GetHumanView()->GetRenderer();
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
                graph->OnRender(tbd::eDRAW_TO_SHADOW_MAP);

                m_drawShadowInstanced->Bind();

                graph->OnRender(tbd::eDRAW_TO_SHADOW_MAP_INSTANCED);
                //d3d::GetContext()->RSSetState(d3d::g_pRasterizerStateFrontFaceSolid);

                graph->PopFrustum();
                renderer->VPopProjectionTransform();

                renderer->ActivateCurrentRendertarget();

                renderer->SetPointLightShadowCubeMapSampler(g_pCubeMapRenderTarget->GetShaderRessourceView());
            
                m_drawLighting->Bind();

                d3d::GetContext()->OMSetBlendState(d3d::g_pBlendStateBlendAdd, NULL, 0xffffff);
                GeometryFactory::GetGlobalScreenQuad()->Bind();
                GeometryFactory::GetGlobalScreenQuad()->Draw();
                d3d::GetContext()->OMSetBlendState(d3d::g_pBlendStateNoBlending, NULL, 0xffffff);

            } break;

        case eDRAW_BOUNDING_DEBUG :
            {
                app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*GetTransformation());
                d3d::ConstBuffer* buffer = app::g_pApp->GetHumanView()->GetRenderer()->GetBuffer(d3d::eBoundingGeoBuffer);
                XMFLOAT4* f  = (XMFLOAT4*)buffer->Map();
                f->x = 1;//m_transformation->GetTransformation()->GetScale().x;
                f->y = 0;
                f->z = 0;
                f->w = 0;
                buffer->Unmap();
                GeometryFactory::GetSphere()->Bind();
                GeometryFactory::GetSphere()->Draw();
            } break;

        case eDRAW_PICKING : 
            {
                util::Mat4 m = *GetTransformation();
                m.SetScale(1);
                DrawPickingSphere(m_actor, &m, 1);
            } break;

        case eDRAW_EDIT_MODE :
            {
                util::Mat4 m = *GetTransformation();
                m.SetScale(1);
                DrawAnchorSphere(m_actor, &m, 1);
            } break;

        case eDRAW_DEBUG_INFOS : 
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
        g_pCubeMapRenderTarget = new d3d::RenderTarget();

        UINT shadowMapSize = app::g_pApp->GetConfig()->GetInteger("iPointLightSMSize");

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