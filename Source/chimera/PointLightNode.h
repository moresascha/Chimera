#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "Frustum.h"
#include "Components.h"

namespace chimera 
{
    class PointlightNode : public SceneNode
    {
    private:
        util::Mat4 m_mats[6];
        util::Mat4 m_projection;
        LightComponent* m_lightComponent;
        PointLightFrustum m_frustum;
        IBlendState* m_alphaBlendingState;
        IDepthStencilState* m_depthState;

        static IShaderProgram* m_drawShadow;
        static IShaderProgram* m_drawShadowInstanced;
        static IShaderProgram* m_drawLighting;
        static std::shared_ptr<IVRamHandle> m_cubeMapHandle;

    public:
        PointlightNode(ActorId actorid);

        void _VRender(ISceneGraph* graph, RenderPath& path);

        bool VIsVisible(ISceneGraph* graph);

        void VOnRestore(ISceneGraph* graph);

        void VOnActorMoved(void);

        ~PointlightNode(void);
    };
};