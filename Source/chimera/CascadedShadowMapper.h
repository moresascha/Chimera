#pragma once
#include "stdafx.h"
#include "RenderTarget.h"
#include "SceneGraph.h"
#include "ShaderProgram.h"
#include "Effect.h"
#include "Event.h"

namespace d3d
{
    struct CascadeSettings
    {
        FLOAT start;
        FLOAT end;
        util::Mat4 m_projection;
    };

    class CascadedShadowMapper
    {
    private:
        d3d::RenderTarget** m_ppTargets;
        d3d::RenderTarget** m_ppBlurredTargets;
        UCHAR m_cascades;
        d3d::ShaderProgram* m_pProgram;
        d3d::ShaderProgram* m_pProgramInstanced;
        d3d::EffectChain** m_ppBlurChain;
        CascadeSettings* m_pCascadesSettings;
        std::shared_ptr<tbd::Actor> m_cascadeCameraActor[3];
        std::shared_ptr<tbd::Actor> m_lightActorCamera;
        std::shared_ptr<tbd::Actor> m_viewActor;
    public:
        CascadedShadowMapper(UCHAR cascades);
        BOOL OnRestore(VOID);
        VOID Render(tbd::SceneGraph* graph);
        VOID Destroy(VOID);
        UCHAR GetSlices(VOID) { return m_cascades; }
        d3d::RenderTarget** GetTargets(VOID) { return m_ppTargets; }

        VOID SetSunPositionDelegate(event::IEventPtr data);

        ~CascadedShadowMapper(VOID);
    };
};
