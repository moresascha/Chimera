#include "SceneGraph.h"
#include "Event.h"
#include "EventManager.h"
#include "GameApp.h"
#include "Actor.h"
#include "Components.h"
#include "SceneNode.h"
#include "Camera.h"
#include "Frustum.h"
namespace tbd 
{

    SceneGraph::SceneGraph() : m_visibiltyReset(TRUE), m_visbibilityCheckTime(0), m_root(std::shared_ptr<ISceneNode>(new SceneNode(INVALID_ACTOR_ID)))
    {
    }

    HRESULT SceneGraph::OnUpdate(ULONG millis) {
        this->m_root->VOnUpdate(millis, this);
        return S_OK;
    }

    HRESULT SceneGraph::OnRender(RenderPath path) {
        //ULONG t = clock();
        TBD_FOR(m_pathToNode[path])
        {
            (*it)->VRender(this, path);
        }
    
        /*if(t > m_visbibilityCheckTime)
        {
            //m_visibiltyReset = FALSE; //wofür war das nochmal?!
        } */
        return S_OK;
    }

    HRESULT SceneGraph::OnRestore(VOID) 
    {
        this->m_root->VOnRestore(this);
        this->m_camera->SetAspect(app::g_pApp->GetWindowWidth(), app::g_pApp->GetWindowHeight());
        return S_OK;
    }

    BOOL SceneGraph::IsVisibilityReset(VOID)
    {
        return m_visibiltyReset;
    }

    VOID SceneGraph::AddChild(ActorId actorId, std::shared_ptr<tbd::ISceneNode> node) {
        this->m_actorMap[actorId] = node;
        this->m_root->VAddChild(node);
        for(UINT i = 0; i <= RENDERPATH_CNT; ++i)
        {
            RenderPath path = 1 << i;
            if(node->VGetRenderPaths() & (path))
            {
                m_pathToNode[path].push_back(node);
            }
        }
    }

    VOID SceneGraph::RemoveChild(ActorId actorid) {
        auto it = m_actorMap.find(actorid);
        if(it == m_actorMap.end())
        {
            LOG_WARNING("actor has no node");
            return;
        }

        std::shared_ptr<ISceneNode> node = it->second;
        for(UINT i = 0; i < RENDERPATH_CNT; ++i)
        {
            RenderPath path = 1 << i;
            if(node->VGetRenderPaths() & path)
            {
                m_pathToNode[path].remove(node);
            }
        }

        this->m_root->VRemoveChild(actorid);
        this->m_actorMap.erase(actorid);
    }

    VOID SceneGraph::SetCamera(std::shared_ptr<util::ICamera> camera)
    {
        this->m_camera = camera;
        m_frustumStack.Clear();
        this->m_camera->SetAspect(app::g_pApp->GetWindowWidth(), app::g_pApp->GetWindowHeight());
    }

    CONST tbd::Frustum* SceneGraph::GetFrustum(VOID)
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

    VOID SceneGraph::PushFrustum(tbd::Frustum* f)
    {
        m_frustumStack.Push(f);
    }

    VOID SceneGraph::PopFrustum(VOID)
    {
        if(m_frustumStack.Size() > 0)
        {
            m_frustumStack.Pop();
        }
    }

    VOID SceneGraph::ResetVisibility(VOID)
    {
        //m_visbibilityCheckTime = clock();
        m_visibiltyReset = TRUE;
    }

    SceneGraph::~SceneGraph(VOID) 
    {

    }
}
