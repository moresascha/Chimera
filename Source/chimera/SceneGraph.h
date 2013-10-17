#pragma once
#include "stdafx.h"
#include "tbdStack.h"
#include "Actor.h"
#include "Vec3.h"

namespace util
{
    class ICamera;
}
namespace chimera 
{
    class ISceneNode;
    class Frustum;

    class SceneGraph : public ISceneGraph
    {
    private:
        std::unique_ptr<ISceneNode> m_root;
        std::shared_ptr<ICamera> m_camera;
        util::tbdStack<Frustum*> m_frustumStack;
        std::map<UINT, std::list<ISceneNode*>> m_pathToNode;
        ULONG m_visbibilityCheckTime;
        BOOL m_visibiltyReset;

    public:
        SceneGraph(VOID);

        VOID VAddChild(ActorId actorid, std::unique_ptr<ISceneNode> child);

        VOID VRemoveChild(ActorId actorid);

        BOOL VOnRender(RenderPath path);

        BOOL VOnRestore(VOID);

        BOOL VOnUpdate(ULONG millis);

        std::shared_ptr<ICamera> VGetCamera(VOID) { return m_camera; }

        CONST Frustum* VGetFrustum(VOID);

        VOID VPushFrustum(Frustum* f);

        VOID VPopFrustum(VOID);

        VOID VResetVisibility(VOID);

        BOOL VIsVisibilityReset(VOID);

        ISceneNode* VFindActorNode(ActorId id);

        VOID VSetCamera(std::shared_ptr<ICamera> camera);

        virtual ~SceneGraph(VOID);
    };
}

