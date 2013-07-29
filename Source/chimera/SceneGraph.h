#pragma once
#include "stdafx.h"
#include "tbdStack.h"
#include "Actor.h"
#include "Vec3.h"
#include "RenderPath.h"

namespace util
{
    class ICamera;
}
namespace tbd 
{
    class ISceneNode;
    class Frustum;

class SceneGraph
{
private:
    std::shared_ptr<tbd::ISceneNode> m_root;
    std::shared_ptr<util::ICamera> m_camera;
    util::tbdStack<tbd::Frustum*> m_frustumStack;
    std::map<UINT, std::list<std::shared_ptr<tbd::ISceneNode>>> m_pathToNode;
    std::map<ActorId, std::shared_ptr<tbd::ISceneNode>> m_actorMap;
    ULONG m_visbibilityCheckTime;
public:
    BOOL m_visibiltyReset;

public:
    SceneGraph(VOID);

    VOID AddChild(ActorId actorid, std::shared_ptr<tbd::ISceneNode>);

    VOID RemoveChild(ActorId actorid);

    HRESULT OnUpdate(ULONG millis);

    std::shared_ptr<util::ICamera> GetCamera(VOID) { return this->m_camera; }

    CONST tbd::Frustum* GetFrustum(VOID);

    VOID PushFrustum(tbd::Frustum* f);

    VOID PopFrustum(VOID);

    VOID ResetVisibility(VOID);

    BOOL IsVisibilityReset(VOID);

    VOID SetCamera(std::shared_ptr<util::ICamera> camera);

    std::shared_ptr<tbd::ISceneNode> RayCast(CONST util::Vec3& position, CONST util::Vec3& direction);

    HRESULT OnRender(RenderPath path);

    HRESULT OnRestore(VOID);

    virtual ~SceneGraph(VOID);
};
}

