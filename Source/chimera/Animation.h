#pragma once
#include "stdafx.h"
#include "SceneNode.h"

namespace tbd
{
    class Animation
    {
    protected:
        std::shared_ptr<tbd::Actor> m_actor;
        util::Vec3 m_translation;
        util::Vec3 m_rotation;
        util::Vec3 m_scale;
    public:
        Animation(std::shared_ptr<tbd::Actor> actor);
        VOID Compute(VOID);
        virtual BOOL TickAnimation(VOID) = 0;
        virtual BOOL VIsDeltaMove(VOID) = 0;
        virtual ~Animation(VOID) {}
    };

    class RotationAnimation : public Animation
    {
    private:

    public:
        virtual BOOL TickAnimation(VOID) = 0;
        virtual BOOL IsDeltaMove(VOID) = 0;
    };

    class AnimationNode : public SceneNode
    {
    public:
        AnimationNode(Animation& animation);
        VOID _VRender(tbd::SceneGraph* graph, tbd::RenderPath& path) {}
        VOID VOnUpdate(ULONG millis, SceneGraph* graph);
    };
}
