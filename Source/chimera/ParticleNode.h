#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "Geometry.h"
#include "Actor.h"
#include "ShaderProgram.h"
#include "ParticleSystem.h"

namespace chimera
{
    class ParticleNode : public SceneNode
    {
    private:
        std::shared_ptr<chimera::ParticleSystem> m_pParticleSystem;
        UINT m_time;
        util::Vec3 m_transformedBBPoint;
        util::Timer m_timer;

        VOID OnFileChanged(VOID);

    public:
        ParticleNode(ActorId id);

        VOID _VRender(chimera::SceneGraph* graph, chimera::RenderPath& path);

        BOOL VIsVisible(SceneGraph* graph);

        VOID VOnRestore(chimera::SceneGraph* graph);

        VOID VOnUpdate(ULONG millis, SceneGraph* graph);

        VOID VOnActorMoved(VOID);

        UINT VGetRenderPaths(VOID);

        ~ParticleNode(VOID);
    };
}


