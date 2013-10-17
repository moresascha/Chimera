#include "CameraTracking.h"
#include "Camera.h"
#include "EventManager.h"
namespace chimera
{
    util::Vec3 LinearInterpolation(CONST util::Vec3& start, CONST util::Vec3& end, FLOAT s)
    {
        return util::Vec3::lerp(start, end, s);
    }

    util::Vec3 QuadtraticInterpolation(CONST util::Vec3& start, CONST util::Vec3& end, FLOAT s)
    {
        return util::Vec3::lerp(start, end, s * s);
    }

    TrackingShot::TrackingShot(std::shared_ptr<chimera::Actor> cameraActor, BOOL repeat) : ActorProcess(cameraActor), 
        m_repeat(repeat), m_time(0)
    {
        m_eyeInterpol = &LinearInterpolation;
        m_focusInterpol = &LinearInterpolation;
        m_pCamer = new util::FPSCamera(10,10,10,10); //cameraActor->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock()->GetCamera();
        m_eyeSpline.SetDivisions(10);
        m_focuspline.SetDivisions(10);
    }

    VOID TrackingShot::AddBasePoint(CONST util::Vec3& eyeStart, CONST util::Vec3& focusStart, UINT millis)
    {
        BasePoint bp;
        bp.eyeStart = eyeStart;
        bp.focusStart = focusStart;
        bp.millis = millis;
        bp.currentTime = 0;
        m_basePoints.push_back(bp);

        m_eyeSpline.AddPoint(eyeStart);
        //m_eyeSpline.AddPoint(eyeEnd);

        m_focuspline.AddPoint(focusStart);
        //m_focuspline.AddPoint(focusEnd);
    }

    VOID TrackingShot::SetDivisions(UINT divs)
    {
        m_eyeSpline.SetDivisions(divs);
        m_focuspline.SetDivisions(divs);
    }

    VOID TrackingShot::SetEyeInterpolation(Interpolation inter)
    {
        m_eyeInterpol = inter;
    }

    VOID TrackingShot::SetFocusInterpolation(Interpolation inter)
    {
        m_focusInterpol = inter;
    }

    VOID TrackingShot::VOnUpdate(ULONG deltaMillis)
    {
        FLOAT s = m_time / (FLOAT)m_animationLength;//m_pCurrentBasePoint->currentTime / (FLOAT)m_pCurrentBasePoint->millis;

        CONST util::Vec3 newEye = m_eyeSpline.GetIntpolPoint(s);//m_eyeInterpol(m_pCurrentBasePoint->eyeStart, m_pCurrentBasePoint->eyeEnd, s);

        CONST util::Vec3 newFocus = m_focuspline.GetIntpolPoint(s);//m_focusInterpol(m_pCurrentBasePoint->focusStart, m_pCurrentBasePoint->focusEnd, s);
        
        m_pCamer->LookAt(newEye, newFocus);

        QUEUE_EVENT(new chimera::MoveActorEvent(m_actor->GetId(), m_pCamer->GetEyePos(), 
            util::Vec3(m_pCamer->GetPhi(), m_pCamer->GetTheta(), 0), FALSE));

        /*m_pCurrentBasePoint->currentTime += deltaMillis;

        if(m_pCurrentBasePoint->currentTime > m_pCurrentBasePoint->millis)
        {
            m_pCurrentBasePoint->currentTime = 0;
            m_pCurrentBasePoint++;
        } */

        m_time += deltaMillis;

        if(m_time > m_animationLength)
        {
            if(!m_repeat)
            {
                Succeed();
            }
            m_time = 0;
        }
    }

    VOID TrackingShot::VOnInit(VOID)
    {
        m_animationLength = 0;
        TBD_FOR(m_basePoints)
        {
            m_animationLength += it->millis;
        }

        m_pCurrentBasePoint = m_basePoints.begin();

        assert(m_basePoints.size() > 0);

        m_eyeSpline.Create();
        m_focuspline.Create();
    }

    TrackingShot::~TrackingShot(VOID)
    {
        SAFE_DELETE(m_pCamer);
    }
}