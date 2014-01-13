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

        VOID _VRender(ISceneGraph* graph, RenderPath& path);

        BOOL VIsVisible(ISceneGraph* graph);

        VOID VOnRestore(ISceneGraph* graph);

        VOID VOnActorMoved(VOID);

        ~PointlightNode(VOID);
    };
};