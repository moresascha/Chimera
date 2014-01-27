#pragma once
#include "stdafx.h"

#include "Mat4.h"
#include "Vec3.h"

namespace chimera
{
    struct CascadeSettings
    {
        float start;
        float end;
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
        util::Vec3 m_intensity;
        util::Vec3 m_ambient;

        void Destroy(void);

    public:

        CascadedShadowMapper(UCHAR cascades);

        bool VOnRestore(void);

        void VRender(ISceneGraph* graph);

        UCHAR VGetSlices(void) { return m_cascades; }

        IRenderTarget** VGetTargets(void) { return m_ppTargets; }

        void SetSunPositionDelegate(IEventPtr data);

        void SetSunIntensityDelegate(IEventPtr data);

        void SetSunAmbientDelegate(IEventPtr data);

        ~CascadedShadowMapper(void);
    };
};
