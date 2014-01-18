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
        std::map<uint, std::list<ISceneNode*>> m_pathToNode;
        ulong m_visbibilityCheckTime;
        bool m_visibiltyReset;

    public:
        SceneGraph(void);

        void VAddChild(ActorId actorid, std::unique_ptr<ISceneNode> child);

        void VRemoveChild(ActorId actorid);

        bool VOnRender(RenderPath path);

        bool VOnRestore(void);

        bool VOnUpdate(ulong millis);

        std::shared_ptr<ICamera> VGetCamera(void) { return m_camera; }

        const Frustum* VGetFrustum(void);

        void VPushFrustum(Frustum* f);

        void VPopFrustum(void);

        void VResetVisibility(void);

        bool VIsVisibilityReset(void);

        ISceneNode* VFindActorNode(ActorId id);

        std::unique_ptr<ISceneNode> VReleaseNode(ActorId id);

        void VSetCamera(std::shared_ptr<ICamera> camera);

        virtual ~SceneGraph(void);
    };
}

