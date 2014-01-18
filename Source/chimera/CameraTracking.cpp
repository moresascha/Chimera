#include "CameraTracking.h"
#include "Camera.h"
#include "EventManager.h"
namespace chimera
{
    util::Vec3 LinearInterpolation(const util::Vec3& start, const util::Vec3& end, float s)
    {
        return util::Vec3::lerp(start, end, s);
    }

    util::Vec3 QuadtraticInterpolation(const util::Vec3& start, const util::Vec3& end, float s)
    {
        return util::Vec3::lerp(start, end, s * s);
    }

    TrackingShot::TrackingShot(std::shared_ptr<chimera::Actor> cameraActor, bool repeat) : ActorProcess(cameraActor), 
        m_repeat(repeat), m_time(0)
    {
        m_eyeInterpol = &LinearInterpolation;
        m_focusInterpol = &LinearInterpolation;
        m_pCamer = new util::FPSCamera(10,10,10,10); //cameraActor->GetComponent<tbd::CameraComponent>(tbd::CameraComponent::COMPONENT_ID).lock()->GetCamera();
        m_eyeSpline.SetDivisions(10);
        m_focuspline.SetDivisions(10);
    }

    void TrackingShot::AddBasePoint(const util::Vec3& eyeStart, const util::Vec3& focusStart, uint millis)
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

    void TrackingShot::SetDivisions(uint divs)
    {
        m_eyeSpline.SetDivisions(divs);
        m_focuspline.SetDivisions(divs);
    }

    void TrackingShot::SetEyeInterpolation(Interpolation inter)
    {
        m_eyeInterpol = inter;
    }

    void TrackingShot::SetFocusInterpolation(Interpolation inter)
    {
        m_focusInterpol = inter;
    }

    void TrackingShot::VOnUpdate(ulong deltaMillis)
    {
        float s = m_time / (float)m_animationLength;//m_pCurrentBasePoint->currentTime / (FLOAT)m_pCurrentBasePoint->millis;

        const util::Vec3 newEye = m_eyeSpline.GetIntpolPoint(s);//m_eyeInterpol(m_pCurrentBasePoint->eyeStart, m_pCurrentBasePoint->eyeEnd, s);

        const util::Vec3 newFocus = m_focuspline.GetIntpolPoint(s);//m_focusInterpol(m_pCurrentBasePoint->focusStart, m_pCurrentBasePoint->focusEnd, s);
        
        m_pCamer->LookAt(newEye, newFocus);

        QUEUE_EVENT(new chimera::MoveActorEvent(m_actor->GetId(), m_pCamer->GetEyePos(), 
            util::Vec3(m_pCamer->GetPhi(), m_pCamer->GetTheta(), 0), false));

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

    void TrackingShot::VOnInit(void)
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

    TrackingShot::~TrackingShot(void)
    {
        SAFE_DELETE(m_pCamer);
    }
}