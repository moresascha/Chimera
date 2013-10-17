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
    SoundProcess::SoundProcess(std::shared_ptr<chimera::ResHandle> handle, INT soundType /* = 0 */, INT volume /* = 100 */, BOOL loop /* = FALSE */)
        : ActorProcess(NULL), m_pHandle(handle), m_volume(volume), m_soundType(soundType), m_loop(loop), m_pSoundBuffer(NULL)
    {

    }

    VOID SoundProcess::VOnInit(VOID)
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

    VOID SoundProcess::VOnAbort(VOID)
    {
        chimera::g_pApp->GetHumanView()->GetSoundSystem()->VReleaseSoundBuffer(m_pSoundBuffer);
    }

    VOID SoundProcess::VOnFail(VOID)
    {
        chimera::g_pApp->GetHumanView()->GetSoundSystem()->VReleaseSoundBuffer(m_pSoundBuffer);
    }

    VOID SoundProcess::VOnSuccess(VOID)
    {
        chimera::g_pApp->GetHumanView()->GetSoundSystem()->VReleaseSoundBuffer(m_pSoundBuffer);
    }

    VOID SoundProcess::VOnUpdate(ULONG deltaMillis)
    {
        if(!m_pSoundBuffer->VIsPlaying())
        {
            Succeed();
        }
    }

    VOID SoundProcess::ComputeVolumeFromDistance(CONST util::Vec3& pos, util::ICamera* camera, FLOAT radius)
    {
        CONST util::Vec3& trans = pos;
        CONST util::Vec3& cameraPos = camera->GetEyePos();
        util::Vec3 sub = trans - cameraPos;
        FLOAT distance = sub.Length();

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
            FLOAT dot = camera->GetSideDir().Dot(sub);

            FLOAT stre = 1 + camera->GetViewDir().Dot(sub);
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
        FLOAT radius,
        INT soundType /* = 0 */, INT volume /* = 100 */, BOOL loop /* = FALSE */)

        : SoundProcess(handle, soundType, volume, loop), m_transform(transCmp), m_radius(radius)
    {
        m_actor = actor;
    }

    VOID SoundEmitterProcess::VOnUpdate(ULONG deltaMillis)
    {
        SoundProcess::VOnUpdate(deltaMillis);
        util::ICamera* camera = chimera::g_pApp->GetHumanView()->GetSceneGraph()->GetCamera().get();
        ComputeVolumeFromDistance(m_transform->GetTransformation()->GetTranslation(), camera, m_radius);
    }

    VOID SoundEmitterProcess::VOnActorDelete(VOID)
    {
        Succeed();
    }

    StaticSoundEmitterProcess::StaticSoundEmitterProcess(CONST util::Vec3& position, std::shared_ptr<chimera::ResHandle> handle, FLOAT radius, INT soundType /* = 0 */, INT volume /* = 100 */, BOOL loop /* = FALSE */)
        :SoundProcess(handle, soundType, volume, loop), m_position(position), m_radius(radius)
    {

    }

    VOID StaticSoundEmitterProcess::VOnInit(VOID)
    {
        SoundProcess::VOnInit();
        util::ICamera* camera = chimera::g_pApp->GetHumanView()->GetSceneGraph()->GetCamera().get();
        ComputeVolumeFromDistance(m_position, camera, m_radius);
    }

    VOID StaticSoundEmitterProcess::VOnUpdate(ULONG deltaMillis)
    {
        SoundProcess::VOnUpdate(deltaMillis);
        util::ICamera* camera = chimera::g_pApp->GetHumanView()->GetSceneGraph()->GetCamera().get();
        ComputeVolumeFromDistance(m_position, camera, m_radius);
    }
}


