#include "Process.h"
#include "GameApp.h"
#include "GameView.h"
#include "Camera.h"
#include "Components.h"
#include "SceneGraph.h"
#include "Sound.h"
#include "math.h"
namespace chimera
{
    SoundProcess::SoundProcess(std::shared_ptr<chimera::ResHandle> handle, int soundType /* = 0 */, int volume /* = 100 */, bool loop /* = FALSE */)
        : ActorProcess(NULL), m_pHandle(handle), m_volume(volume), m_soundType(soundType), m_loop(loop), m_pSoundBuffer(NULL)
    {

    }

    void SoundProcess::VOnInit(void)
    {
        ActorProcess::VOnInit();

        if(!m_pHandle->IsReady())
        {
            m_pHandle = chimera::g_pApp->GetCache()->GetHandle(m_pHandle->GetResource());
        }
        m_pSoundBuffer = chimera::g_pApp->GetHumanView()->GetSoundSystem()->VCreateSoundBuffer(m_pHandle);
        m_pSoundBuffer->VSetVolume(m_volume);
        m_pSoundBuffer->VPlay(m_loop);
    }

    void SoundProcess::VOnAbort(void)
    {
        chimera::g_pApp->GetHumanView()->GetSoundSystem()->VReleaseSoundBuffer(m_pSoundBuffer);
    }

    void SoundProcess::VOnFail(void)
    {
        chimera::g_pApp->GetHumanView()->GetSoundSystem()->VReleaseSoundBuffer(m_pSoundBuffer);
    }

    void SoundProcess::VOnSuccess(void)
    {
        chimera::g_pApp->GetHumanView()->GetSoundSystem()->VReleaseSoundBuffer(m_pSoundBuffer);
    }

    void SoundProcess::VOnUpdate(ulong deltaMillis)
    {
        if(!m_pSoundBuffer->VIsPlaying())
        {
            Succeed();
        }
    }

    void SoundProcess::ComputeVolumeFromDistance(const util::Vec3& pos, util::ICamera* camera, float radius)
    {
        const util::Vec3& trans = pos;
        const util::Vec3& cameraPos = camera->GetEyePos();
        util::Vec3 sub = trans - cameraPos;
        float distance = sub.Length();

        if(distance > radius)
        {
            if(m_pSoundBuffer->VGetVolume() > -10000) //todo
            {
                m_pSoundBuffer->VSetVolume(0);
            }
        }
        else
        {
            LONG vol = 100;
            sub.Normalize();
            float dot = camera->GetSideDir().Dot(sub);

            float stre = 1 + camera->GetViewDir().Dot(sub);
            stre = CLAMP(stre, 0.8f, 1.0f);

            //DEBUG_OUT_A("%f %f", dot, stre);

            m_pSoundBuffer->VSetPan(dot);
            
            vol = (LONG) (vol * stre);
            
            if(distance > radius * 0.5)
            {
                vol = (LONG)(stre * 100.0f * (1.0f - 2 * (distance / radius - 0.5f)));
            }
            m_pSoundBuffer->VSetVolume(vol);
        }
    }

    SoundEmitterProcess::SoundEmitterProcess(
        std::shared_ptr<chimera::Actor> actor,
        std::shared_ptr<chimera::TransformComponent> transCmp, 
        std::shared_ptr<chimera::ResHandle> handle,
        float radius,
        int soundType /* = 0 */, int volume /* = 100 */, bool loop /* = FALSE */)

        : SoundProcess(handle, soundType, volume, loop), m_transform(transCmp), m_radius(radius)
    {
        m_actor = actor;
    }

    void SoundEmitterProcess::VOnUpdate(ulong deltaMillis)
    {
        SoundProcess::VOnUpdate(deltaMillis);
        util::ICamera* camera = chimera::g_pApp->GetHumanView()->GetSceneGraph()->GetCamera().get();
        ComputeVolumeFromDistance(m_transform->GetTransformation()->GetTranslation(), camera, m_radius);
    }

    void SoundEmitterProcess::VOnActorDelete(void)
    {
        Succeed();
    }

    StaticSoundEmitterProcess::StaticSoundEmitterProcess(const util::Vec3& position, std::shared_ptr<chimera::ResHandle> handle, float radius, int soundType /* = 0 */, int volume /* = 100 */, bool loop /* = FALSE */)
        :SoundProcess(handle, soundType, volume, loop), m_position(position), m_radius(radius)
    {

    }

    void StaticSoundEmitterProcess::VOnInit(void)
    {
        SoundProcess::VOnInit();
        util::ICamera* camera = chimera::g_pApp->GetHumanView()->GetSceneGraph()->GetCamera().get();
        ComputeVolumeFromDistance(m_position, camera, m_radius);
    }

    void StaticSoundEmitterProcess::VOnUpdate(ulong deltaMillis)
    {
        SoundProcess::VOnUpdate(deltaMillis);
        util::ICamera* camera = chimera::g_pApp->GetHumanView()->GetSceneGraph()->GetCamera().get();
        ComputeVolumeFromDistance(m_position, camera, m_radius);
    }
}


