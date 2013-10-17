#pragma once
#include "stdafx.h"
#include "Process.h"
#include "Vec3.h"
#include "Spline.h"

namespace util
{
    class ICamera;
}

namespace chimera
{
    typedef util::Vec3 (*Interpolation)(CONST util::Vec3& v0, CONST util::Vec3& v1, FLOAT s);

    struct BasePoint
    {
        util::Vec3 eyeStart;
        util::Vec3 eyeEnd;

        util::Vec3 focusStart;
        util::Vec3 focusEnd;

        UINT millis;

        UINT currentTime;
    };

    class TrackingShot : public ActorProcess
    {
    private:
        std::vector<BasePoint> m_basePoints;
        util::Vec3 m_currentEye;
        UINT m_animationLength; //millis
        UINT m_time;
        chimera::ICamera* m_pCamer;
        BOOL m_repeat;
        std::vector<BasePoint>::iterator m_pCurrentBasePoint;
        util::UniformBSpline m_eyeSpline;
        util::UniformBSpline m_focuspline;

        Interpolation m_eyeInterpol;
        Interpolation m_focusInterpol;

    public:
        TrackingShot(std::shared_ptr<chimera::Actor> cameraActor, BOOL repeat = FALSE);
        VOID AddBasePoint(CONST util::Vec3& eyeStart, CONST util::Vec3& focusStart, UINT millis = 5000);
        VOID VOnInit(VOID);
        VOID VOnUpdate(ULONG deltaMillis);
        VOID SetDivisions(UINT divs);
        VOID SetEyeInterpolation(Interpolation inter);
        VOID SetFocusInterpolation(Interpolation inter);
        ~TrackingShot(VOID);
    };
}
