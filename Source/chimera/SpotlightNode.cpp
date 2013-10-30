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

namespace chimera
{
    chimera::RenderTarget* SpotlightNode::g_pShadowRenderTarget = NULL;

    SpotlightNode::SpotlightNode(ActorId actorid) : SceneNode(actorid), m_distance(0)
    {
        std::shared_ptr<chimera::Actor> actor = chimera::g_pApp->GetLogic()->VFindActor(actorid);
        this->m_lightComponent = m_actor->GetComponent<chimera::LightComponent>(chimera::LightComponent::COMPONENT_ID).lock();

        m_drawShadow = chimera::ShaderProgram::GetProgram("SpotLightShadowMap").get();
        m_drawShadowInstanced = chimera::ShaderProgram::GetProgram("SpotLightShadowMapInstanced").get();
        m_drawLighting = chimera::ShaderProgram::GetProgram("SpotLight").get();

        UINT wh = chimera::g_pApp->GetConfig()->GetInteger("iSpotLightSMSize");

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

    VOID SpotlightNode::VOnRestore(chimera::SceneGraph* graph)
    {
        SceneNode::VOnRestore(graph);
    }

    BOOL SpotlightNode::VIsVisible(SceneGraph* graph)
    {
        return graph->GetFrustum()->IsInside(m_middle,  m_distance);
    }

    UINT SpotlightNode::VGetRenderPaths(VOID)
    {
        return eRenderPath_DrawLighting | eRenderPath_DrawPicking | eRenderPath_DrawBounding | eRenderPath_DrawEditMode | eRenderPath_DrawDebugInfo;
    }

    VOID SpotlightNode::_VRender(chimera::SceneGraph* graph, RenderPath& path)
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
                renderer->VPushViewTransform(m_pCamera->GetView(), m_pCamera->GetIView(), m_pCamera->GetEyePos());
                renderer->VPushProjectionTransform(m_pCamera->GetProjection(), GetTransformation()->GetScale().x);

                renderer->SetLightSettings(m_lightComponent->m_color, GetTransformation()->GetTranslation(), 
                    m_pCamera->GetViewDir(),
                    GetTransformation()->GetScale().x, DEGREE_TO_RAD(m_lightComponent->m_angle), m_lightComponent->m_intensity);

                g_pShadowRenderTarget->Bind();
                g_pShadowRenderTarget->Clear();

                graph->PushFrustum(&m_pCamera->GetFrustum());

                graph->OnRender(chimera::eRenderPath_DrawToShadowMap);

                m_drawShadowInstanced->Bind();

                graph->OnRender(chimera::eRenderPath_DrawToShadowMapInstanced);

                graph->PopFrustum();

                renderer->ActivateCurrentRendertarget();

                renderer->SetSampler(chimera::eDiffuseColorSampler, g_pShadowRenderTarget->GetShaderRessourceView()); //todo

                m_drawLighting->Bind();

                chimera::GetContext()->OMSetDepthStencilState(chimera::m_pNoDepthNoStencilState, 0);
                chimera::GetContext()->OMSetBlendState(chimera::g_pBlendStateBlendAdd, NULL, 0xffffff);
                GeometryFactory::GetGlobalScreenQuad()->Bind();
                GeometryFactory::GetGlobalScreenQuad()->Draw();
                chimera::GetContext()->OMSetBlendState(chimera::g_pBlendStateNoBlending, NULL, 0xffffff);
                chimera::GetContext()->OMSetDepthStencilState(chimera::m_pDepthNoStencilState, 0);

                renderer->VPopViewTransform();
                renderer->VPopProjectionTransform();

                renderer->SetSampler(chimera::eDiffuseColorSampler, NULL); //todo

            } break;

        case eRenderPath_DrawBounding :
            {
                util::Mat4 t;
                t.SetTranslate(m_middle.x, m_middle.y, m_middle.z);
                DrawSphere(&t, m_distance);
            } break;

        case eRenderPath_DrawPicking : 
            {
                util::Mat4 m = *GetTransformation();
                m.SetScale(1);
                DrawPickingSphere(m_actor, &m, 1);
            } break;

        case eRenderPath_DrawEditMode :
            {
                if(!HasParent())
                {
                    util::Mat4 m = *GetTransformation();
                    m.SetScale(1);
                    DrawAnchorSphere(m_actor, &m, 1); 
                   // DrawFrustum(m_pCamera->GetFrustum());
                }
            } break;

        case eRenderPath_DrawDebugInfo : 
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
        g_pShadowRenderTarget = new chimera::RenderTarget();

        UINT shadowMapSize = chimera::g_pApp->GetConfig()->GetInteger("iSpotLightSMSize");

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