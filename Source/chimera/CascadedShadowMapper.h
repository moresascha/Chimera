#pragma once
#include "stdafx.h"

#include "Mat4.h"
#include "Vec3.h"

namespace chimera
{
    struct CascadeSettings
    {
        FLOAT start;
        FLOAT end;
        util::Mat4 m_projection;
    };

    class CascadedShadowMapper : public IEnvironmentLighting
    {
    private:
        IRenderTarget** m_ppTargets;
        IRenderTarget** m_ppBlurredTargets;
        UCHAR m_cascades;
        IShaderProgram* m_pProgram;
        IShaderProgram* m_pProgramInstanced;
        IEffectChain** m_ppBlurChain;
        CascadeSettings* m_pCascadesSettings;
        IActor* m_cascadeCameraActor[3];
        IActor* m_lightActorCamera;
        IActor* m_viewActor;

        VOID Destroy(VOID);

    public:

        CascadedShadowMapper(UCHAR cascades);

        BOOL VOnRestore(VOID);

        VOID VRender(ISceneGraph* graph);

        UCHAR VGetSlices(VOID) { return m_cascades; }

        IRenderTarget** VGetTargets(VOID) { return m_ppTargets; }

        VOID SetSunPositionDelegate(IEventPtr data);

        ~CascadedShadowMapper(VOID);
    };
};
