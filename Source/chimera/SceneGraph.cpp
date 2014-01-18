#include "SceneGraph.h"
#include "Event.h"
#include "SceneNode.h"

namespace chimera 
{
    SceneGraph::SceneGraph() : m_visibiltyReset(true), m_visbibilityCheckTime(0), m_root(std::unique_ptr<ISceneNode>(new SceneNode(CM_INVALID_ACTOR_ID)))
    {
    }

    bool SceneGraph::VOnUpdate(ulong millis) 
    {
        m_root->VOnUpdate(millis, this);
        return true;
    }

    bool SceneGraph::VOnRender(RenderPath path) 
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
        return true;
    }

    bool SceneGraph::VOnRestore(void) 
    {
        m_root->VOnRestore(this);
        if(m_camera)
        {
            m_camera->SetAspect(chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetWidth(), chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetHeight());
        }
        return true;
    }

    bool SceneGraph::VIsVisibilityReset(void)
    {
        return m_visibiltyReset;
    }

    void SceneGraph::VAddChild(ActorId actorId, std::unique_ptr<ISceneNode> node) 
    {
        for(uint i = 0; i <= CM_RENDERPATH_CNT; ++i)
        {
            RenderPath path = (RenderPath)(1 << i);
            if(node->VGetRenderPaths() & (path))
            {
                m_pathToNode[path].push_back(node.get());
            }
        }
        
        m_root->VAddChild(std::move(node));
    }

    std::unique_ptr<ISceneNode> SceneGraph::VReleaseNode(ActorId id)
    {
        return m_root->VRemoveChild(id);
    }

    void SceneGraph::VRemoveChild(ActorId actorid) {

        ISceneNode* node = m_root->VFindActor(actorid);

        if(!node)
        {
            LOG_CRITICAL_ERROR("No actor node found");
        }

        for(uint i = 0; i < CM_RENDERPATH_CNT; ++i)
        {
            RenderPath path = (RenderPath)(1 << i);
            if(node->VGetRenderPaths() & path)
            {
                m_pathToNode[path].remove(node);
            }
        }

        m_root->VRemoveChild(actorid);
    }

    void SceneGraph::VSetCamera(std::shared_ptr<ICamera> camera)
    {
        m_camera = camera;
        m_frustumStack.Clear();
        m_camera->SetAspect(chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetWidth(), chimera::CmGetApp()->VGetHumanView()->VGetRenderer()->VGetHeight());
        m_root->VForceVisibilityCheck();
    }

    const Frustum* SceneGraph::VGetFrustum(void)
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

    void SceneGraph::VPushFrustum(Frustum* f)
    {
        m_frustumStack.Push(f);
    }

    void SceneGraph::VPopFrustum(void)
    {
        if(m_frustumStack.Size() > 0)
        {
            m_frustumStack.Pop();
        }
    }

    void SceneGraph::VResetVisibility(void)
    {
        //m_visbibilityCheckTime = clock();
        m_visibiltyReset = true;
    }

    ISceneNode* SceneGraph::VFindActorNode(ActorId id)
    {
        return m_root->VFindActor(id);
    }

    SceneGraph::~SceneGraph(void) 
    {

    }
}
