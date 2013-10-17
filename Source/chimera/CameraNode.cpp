#include "SceneNode.h"
#include "Components.h"
#include "Camera.h"
#include "SceneGraph.h"
namespace chimera
{
    CameraNode::CameraNode(ActorId id) : SceneNode(id)
    {
        m_pCamera = m_actor->GetComponent<chimera::CameraComponent>(chimera::CameraComponent::COMPONENT_ID).lock().get()->GetCamera();
    }

    VOID CameraNode::_VRender(chimera::SceneGraph* graph, chimera::RenderPath& path)
    {
        if(m_pCamera != graph->GetCamera())
        {
            DrawFrustum(m_pCamera->GetFrustum());
        }
    }

    UINT CameraNode::VGetRenderPaths(VOID)
    {
        return chimera::eRenderPath_NoDraw;
    }
}
