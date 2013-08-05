#include "SceneNode.h"
#include "Camera.h"
#include "d3d.h"
#include "GameApp.h"
#include "SceneGraph.h"
#include "D3DRenderer.h"
#include "Components.h"
#include "Geometry.h"
#include "Mesh.h"
#include "GeometryFactory.h"
namespace tbd
{
    SkyDomeNode::SkyDomeNode(ActorId id, tbd::Resource res) : SceneNode(id), m_TextureRes(res)
    {

    }

    BOOL SkyDomeNode::VIsVisible(SceneGraph* graph)
    {
        return TRUE;
    }

    VOID SkyDomeNode::VOnRestore(tbd::SceneGraph* graph)
    {
        m_actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock()->GetTransformation()->SetScale(400);
        m_textureHandle = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_TextureRes));
    }

    VOID SkyDomeNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        switch(path)
        {
        case eDRAW_SKY :
            {
                if(m_textureHandle->IsReady())
                {
                    m_textureHandle->Update();
                    util::ICamera* cam = graph->GetCamera().get();
                    tbd::TransformComponent* tc = m_actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock().get();
                    tc->GetTransformation()->SetTranslate(cam->GetEyePos().x, cam->GetEyePos().y, cam->GetEyePos().z);
                    app::g_pApp->GetHumanView()->GetRenderer()->VPushWorldTransform(*tc->GetTransformation());
                    app::g_pApp->GetHumanView()->GetRenderer()->SetSampler(d3d::eDiffuseColorSampler, m_textureHandle->GetShaderResourceView());
                    GeometryFactory::GetSkyDome()->Bind();
                    GeometryFactory::GetSkyDome()->Draw();
                }
                else
                {
                    m_textureHandle = std::static_pointer_cast<d3d::Texture2D>(app::g_pApp->GetHumanView()->GetVRamManager()->GetHandle(m_TextureRes));
                }
            } break;

        case eDRAW_BOUNDING_DEBUG :
            {
                tbd::TransformComponent* tc = m_actor->GetComponent<tbd::TransformComponent>(tbd::TransformComponent::COMPONENT_ID).lock().get();
                DrawSphere(tc->GetTransformation(), 1);
            } break;
        }

    }

    RenderPath SkyDomeNode::VGetRenderPaths(VOID)
    {
        return eDRAW_SKY | eDRAW_BOUNDING_DEBUG;
    }
}


