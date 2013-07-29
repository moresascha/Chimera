#include "SceneNode.h"
#include "SceneGraph.h"
#include "Components.h"
#include "Camera.h"
namespace tbd
{
    CameraDebugNode::CameraDebugNode(ActorId id) : SceneNode(id)
    {
        m_pCamera = m_actor->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock().get()->GetCamera();
    }

    VOID CameraDebugNode::_VRender(tbd::SceneGraph* graph, tbd::RenderPath& path)
    {
        if(m_pCamera != graph->GetCamera() /*&& m_actor->GetName() == "cascadeLightCamera" */)
        {
            DrawFrustum(m_pCamera->GetFrustum());
        }
    }

    UINT CameraDebugNode::VGetRenderPaths(VOID)
    {
        return eDRAW_TO_ALBEDO;
    }
}
