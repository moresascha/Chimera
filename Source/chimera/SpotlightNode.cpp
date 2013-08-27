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
#include "math.h"

namespace tbd
{
    d3d::RenderTarget* SpotlightNode::g_pShadowRenderTarget = NULL;

    SpotlightNode::SpotlightNode(ActorId actorid) : SceneNode(actorid), m_distance(0)
    {
        std::shared_ptr<tbd::Actor> actor = app::g_pApp->GetLogic()->VFindActor(actorid);
        this->m_lightComponent = m_actor->GetComponent<tbd::LightComponent>(tbd::LightComponent::COMPONENT_ID).lock();

        m_drawShadow = d3d::ShaderProgram::GetProgram("SpotLightShadowMap").get();
        m_drawShadowInstanced = d3d::ShaderProgram::GetProgram("SpotLightShadowMapInstanced").get();
        m_drawLighting = d3d::ShaderProgram::GetProgram("SpotLight").get();

        UINT wh = app::g_pApp->GetConfig()->GetInteger("iSpotLightSMSize");

        m_pCamera = new util::Camera(wh, wh, 1e-2f, GetTransformation()->GetScale().x);
        
        VOnActorMoved();
    }

    VOID SpotlightNode::VOnActorMoved(VOID)
    {
        SceneNode::VOnActorMoved();
        m_pCamera->SetPerspectiveProjection(1.0f, DEGREE_TO_RAD(m_lightComponent->m_angle), 0.01f, GetTransformation()->GetScale().x);
        
        util::Vec4 up(0,1,0,0);
        util::Vec4 dir(0,0,1,0);
        up = util::Mat4::Transform(*GetTransformation(), up);
        dir = util::Mat4::Transform(*GetTransformation(), dir);
        up.Normalize();
        dir.Normalize();
        m_pCamera->FromViewUp(util::Vec3(dir.x,dir.y,dir.z), util::Vec3(up.x,up.y,up.z));

        m_pCamera->SetEyePos(GetTransformation()->GetTranslation());

        //m_pCamera->SetRotation(GetTransformation()->GetPYR().y, GetTransformation()->GetPYR().x);
        
        m_middle = GetTransformation()->GetTranslation();
        FLOAT c = cos(m_pCamera->GetFoV() / 2.0f); //Todo: need a tighter bb here
        FLOAT h = GetTransformation()->GetScale().x / c;
        m_distance = h / 2.0f;
        m_middle = m_middle + (m_pCamera->GetViewDir() * m_distance);
    }

    VOID SpotlightNode::VOnRestore(tbd::SceneGraph* graph)
    {
        SceneNode::VOnRestore(graph);
    }

    BOOL SpotlightNode::VIsVisible(SceneGraph* graph)
    {
        return graph->GetFrustum()->IsInside(m_middle,  m_distance);
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
                renderer->VPushProjectionTransform(m_pCamera->GetProjection(), GetTransformation()->GetScale().x);

				renderer->SetLightSettings(m_lightComponent->m_color, GetTransformation()->GetTranslation(), 
                    m_pCamera->GetViewDir(),
                    GetTransformation()->GetScale().x, DEGREE_TO_RAD(m_lightComponent->m_angle), m_lightComponent->m_intensity);

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

                d3d::GetContext()->OMSetDepthStencilState(d3d::m_pNoDepthNoStencilState, 0);
                d3d::GetContext()->OMSetBlendState(d3d::g_pBlendStateBlendAdd, NULL, 0xffffff);
                GeometryFactory::GetGlobalScreenQuad()->Bind();
                GeometryFactory::GetGlobalScreenQuad()->Draw();
                d3d::GetContext()->OMSetBlendState(d3d::g_pBlendStateNoBlending, NULL, 0xffffff);
                d3d::GetContext()->OMSetDepthStencilState(d3d::m_pDepthNoStencilState, 0);

                renderer->VPopViewTransform();
                renderer->VPopProjectionTransform();

                renderer->SetSampler(d3d::eDiffuseColorSampler, NULL); //todo

            } break;

        case eDRAW_BOUNDING_DEBUG :
            {
                util::Mat4 t;
                t.SetTranslate(m_middle.x, m_middle.y, m_middle.z);
                DrawSphere(&t, m_distance);
            } break;

        case eDRAW_PICKING : 
            {
                util::Mat4 m = *GetTransformation();
                m.SetScale(1);
                DrawPickingSphere(m_actor, &m, 1);
            } break;

        case eDRAW_EDIT_MODE :
            {
                if(!HasParent())
                {
                    util::Mat4 m = *GetTransformation();
                    m.SetScale(1);
                    DrawAnchorSphere(m_actor, &m, 1); 
                   // DrawFrustum(m_pCamera->GetFrustum());
                }
            } break;

        case eDRAW_DEBUG_INFOS : 
            {            
                std::stringstream ss;
                ss << "SpotLight_";
                ss << m_actorId;
                DrawInfoTextOnScreen(graph->GetCamera().get(), GetTransformation(), ss.str());
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

        UINT shadowMapSize = app::g_pApp->GetConfig()->GetInteger("iSpotLightSMSize");

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