#include "SpotlightNode.h"
#include "Vec3.h"
#include "Mat4.h"
#include "SceneGraph.h"
#include "ShaderProgram.h"
#include "GameApp.h"
#include "GeometryFactory.h"
#include "Components.h"
#include "D3DRenderer.h"
#include "GameLogic.h"
#include "Camera.h"

namespace tbd
{
    d3d::RenderTarget* SpotlightNode::g_pShadowRenderTarget = NULL;

    SpotlightNode::SpotlightNode(ActorId actorid) : SceneNode(actorid)
    {
        std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VFindActor(actorid);
        this->m_lightComponent = m_actor->GetComponent<tbd::LightComponent>(tbd::LightComponent::COMPONENT_ID).lock();

        m_drawShadow = d3d::ShaderProgram::GetProgram("SpotLightShadowMap").get();
        m_drawShadowInstanced = d3d::ShaderProgram::GetProgram("SpotLightShadowMapInstanced").get();
        m_drawLighting = d3d::ShaderProgram::GetProgram("SpotLight").get();

        m_pCamera = new util::FPSCamera(512, 512, 1e-2f, m_transformation->GetTransformation()->GetScale().x);
        
        VOnActorMoved();
    }

    VOID SpotlightNode::VOnActorMoved(VOID)
    {
        SceneNode::VOnActorMoved();
        m_pCamera->SetPerspectiveProjection(1.0f, XM_PIDIV2, 0.01f, m_transformation->GetTransformation()->GetScale().x);
        
        m_pCamera->LookAt(m_transformation->GetTransformation()->GetTranslation(), util::Vec3(0,0,0));
        /*m_pCamera->SetEyePos(m_transformation->GetTransformation()->GetTranslation());
        m_pCamera->SetRotation(m_transformation->GetTransformation()->GetRotation().x, m_transformation->GetTransformation()->GetRotation().y); */
    }

    VOID SpotlightNode::VOnRestore(tbd::SceneGraph* graph)
    {
        SceneNode::VOnRestore(graph);
    }

    BOOL SpotlightNode::VIsVisible(SceneGraph* graph)
    {
        return TRUE;//graph->GetFrustum()->IsInside(m_transformation->GetTransformation()->GetTranslation(),  m_transformation->GetTransformation()->GetScale().x);
    }

    UINT SpotlightNode::VGetRenderPaths(VOID)
    {
        return eDRAW_LIGHTING | eDRAW_PICKING | eDRAW_BOUNDING_DEBUG | eDRAW_EDIT_MODE | eDRAW_DEBUG_INFOS;
    }

    VOID SpotlightNode::_VRender(tbd::SceneGraph* graph, RenderPath& path)
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
                renderer->VPushViewTransform(m_pCamera->GetView(), m_pCamera->GetIView(), m_pCamera->GetEyePos());
                renderer->VPushProjectionTransform(m_pCamera->GetProjection(), m_transformation->GetTransformation()->GetScale().x);

				renderer->SetLightSettings(m_lightComponent->m_color, m_transformation->GetTransformation()->GetTranslation(), m_transformation->GetTransformation()->GetScale().x);

                g_pShadowRenderTarget->Bind();
                g_pShadowRenderTarget->Clear();

                graph->PushFrustum(&m_pCamera->GetFrustum());

                graph->OnRender(tbd::eDRAW_TO_SHADOW_MAP);

                m_drawShadowInstanced->Bind();

                graph->OnRender(tbd::eDRAW_TO_SHADOW_MAP_INSTANCED);

                graph->PopFrustum();

                renderer->ActivateCurrentRendertarget();

                renderer->SetSampler(d3d::eDiffuseColorSampler, g_pShadowRenderTarget->GetShaderRessourceView()); //todo

                m_drawLighting->Bind();

                d3d::GetContext()->OMSetBlendState(d3d::g_pBlendStateBlendAdd, NULL, 0xffffff);
                GeometryFactory::GetGlobalScreenQuad()->Bind();
                GeometryFactory::GetGlobalScreenQuad()->Draw();
                d3d::GetContext()->OMSetBlendState(d3d::g_pBlendStateNoBlending, NULL, 0xffffff);

                renderer->VPopViewTransform();
                renderer->VPopProjectionTransform();

            } break;

        case eDRAW_BOUNDING_DEBUG :
            {
                app::g_pApp->GetHumanView()->GetRenderer()->SetNormalMapping(FALSE);
                app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*m_transformation->GetTransformation());
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
                util::Mat4 m = *m_transformation->GetTransformation();
                m.SetScale(1);
                DrawPickingSphere(m_actor, &m, 1);
            } break;

        case eDRAW_EDIT_MODE :
            {
                util::Mat4 m = *m_transformation->GetTransformation();
                m.SetScale(1);
                DrawAnchorSphere(m_actor, &m, 1);
                DrawFrustum(m_pCamera->GetFrustum());
            } break;

        case eDRAW_DEBUG_INFOS : 
            {            
                std::stringstream ss;
                ss << "SpotLight_";
                ss << m_actorId;
                DrawInfoTextOnScreen(graph->GetCamera().get(), m_transformation->GetTransformation(), ss.str());
                break;
            }
        }
    }

    SpotlightNode::~SpotlightNode(VOID)
    {
        SAFE_DELETE(m_pCamera);
    }

    //static

    BOOL SpotlightNode::Create(VOID)
    {
        g_pShadowRenderTarget = new d3d::RenderTarget();

        UINT shadowMapSize = 512;//app::g_pApp->GetConfig()->GetInteger("iPointLightSMSize");

        if(!g_pShadowRenderTarget->OnRestore(shadowMapSize, shadowMapSize, DXGI_FORMAT_R32_FLOAT, TRUE))
        {
            LOG_CRITICAL_ERROR("Failed to create render target");
            return FALSE;
        }

        return TRUE;
    }

    VOID SpotlightNode::Destroy(VOID)
    {
        SAFE_DELETE(g_pShadowRenderTarget);
    }
}