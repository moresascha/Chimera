#pragma once
#include "stdafx.h"
#include "SceneNode.h"
#include "Mat4.h"
#include "Frustum.h"

namespace chimera
{
    class LightComponent;
    class SpotlightNode : public chimera::SceneNode
    {
    private:
        chimera::ICamera* m_pCamera;
        std::shared_ptr<LightComponent> m_lightComponent;
        util::Vec3 m_middle;
        float m_distance;
        IBlendState* m_alphaBlendingState;
        IDepthStencilState* m_depthState;
        std::shared_ptr<IDeviceTexture> m_projectedTextureHandle;

        static std::shared_ptr<IVRamHandle> m_pShadowRenderTargetHandle;
        static IShaderProgram* m_drawShadow;
        static IShaderProgram* m_drawShadowInstanced;
        static IShaderProgram* m_drawLighting;

    public:
        SpotlightNode(ActorId actorid);

        void _VRender(ISceneGraph* graph, RenderPath& path);

        bool VIsVisible(ISceneGraph* graph);

        void VOnRestore(ISceneGraph* graph);

        void VOnActorMoved(void);

        ~SpotlightNode(void);
    };
}
