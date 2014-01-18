#include "SceneNode.h"
#include "Components.h"
#include "Camera.h"
#include "SceneGraph.h"
namespace chimera
{
    CameraNode::CameraNode(ActorId id) : SceneNode(id)
    {
        CameraComponent* cmp;
        m_actor->VQueryComponent(CM_CMP_CAMERA, (IActorComponent**)(&cmp));
        m_pCamera = cmp->m_camera;
        VSetRenderPaths(0);
    }

    void CameraNode::_VRender(ISceneGraph* graph, RenderPath& path)
    {
        /*if(m_pCamera != graph->GetCamera())
        {
            VDrawFrustum(m_pCamera->GetFrustum());
        }*/
    }
}
