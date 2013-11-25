#include "SceneGraph.h"
#include "Event.h"
#include "SceneNode.h"

namespace chimera 
{
    SceneGraph::SceneGraph() : m_visibiltyReset(TRUE), m_visbibilityCheckTime(0), m_root(std::unique_ptr<ISceneNode>(new SceneNode(CM_INVALID_ACTOR_ID)))
    {
    }

    BOOL SceneGraph::VOnUpdate(ULONG millis) 
    {
        m_root->VOnUpdate(millis, this);
        return TRUE;
    }

    BOOL SceneGraph::VOnRender(RenderPath path) 
    {
        //ULONG t = clock();
        TBD_FOR(m_pathToNode[path])
        {
            (*it)->VRender(this, path);
        }
    
        /*if(t > m_visbibilityCheckTime)
        {
            //m_visibiltyReset = FALSE; //wofür war das nochmal?!
        } */
        return TRUE;
    }

    BOOL SceneGraph::VOnRestore(VOID) 
    {
        m_root->VOnRestore(this);
        if(m_camera)
        {
            m_camera->SetAspect(chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetWidth(), chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetHeight());
        }
        return TRUE;
    }

    BOOL SceneGraph::VIsVisibilityReset(VOID)
    {
        return m_visibiltyReset;
    }

    VOID SceneGraph::VAddChild(ActorId actorId, std::unique_ptr<ISceneNode> node) 
    {
        for(UINT i = 0; i <= CM_RENDERPATH_CNT; ++i)
        {
            RenderPath path = (RenderPath)(1 << i);
            if(node->VGetRenderPaths() & (path))
            {
                m_pathToNode[path].push_back(node.get());
            }
        }
        
        m_root->VAddChild(std::move(node));
    }

    VOID SceneGraph::VRemoveChild(ActorId actorid) {

        ISceneNode* node = m_root->VFindActor(actorid);

        if(!node)
        {
            LOG_CRITICAL_ERROR("No actor node found");
        }

        for(UINT i = 0; i < CM_RENDERPATH_CNT; ++i)
        {
            RenderPath path = (RenderPath)(1 << i);
            if(node->VGetRenderPaths() & path)
            {
                m_pathToNode[path].remove(node);
            }
        }

        m_root->VRemoveChild(actorid);
    }

    VOID SceneGraph::VSetCamera(std::shared_ptr<ICamera> camera)
    {
        m_camera = camera;
        m_frustumStack.Clear();
        m_camera->SetAspect(chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetWidth(), chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetHeight());
        m_root->VForceVisibilityCheck();
    }

    CONST Frustum* SceneGraph::VGetFrustum(VOID)
    { 
        if(m_frustumStack.Size() == 0)
        {
            return &m_camera->GetFrustum();
        }
        else
        {
            return m_frustumStack.Peek(); 
        }
    }

    VOID SceneGraph::VPushFrustum(Frustum* f)
    {
        m_frustumStack.Push(f);
    }

    VOID SceneGraph::VPopFrustum(VOID)
    {
        if(m_frustumStack.Size() > 0)
        {
            m_frustumStack.Pop();
        }
    }

    VOID SceneGraph::VResetVisibility(VOID)
    {
        //m_visbibilityCheckTime = clock();
        m_visibiltyReset = TRUE;
    }

    ISceneNode* SceneGraph::VFindActorNode(ActorId id)
    {
        return m_root->VFindActor(id);
    }

    SceneGraph::~SceneGraph(VOID) 
    {

    }
}
