#include "SceneNode.h"
#include "Components.h"
#include "Camera.h"
#include "SceneGraph.h"
namespace tbd
{
    CameraNode::CameraNode(ActorId id) : SceneNode(id)
    {
        m_pCamera = m_actor->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock().get()->GetCamera();
    }

    VOID CameraNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        if(m_pCamera != graph->GetCamera())
        {
            DrawFrustum(m_pCamera->GetFrustum());
        }
    }

    UINT CameraNode::VGetRenderPaths(VOID)
    {
        return tbd::eDRAW_EDIT_MODE;
    }
}
