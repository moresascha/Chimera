#pragma once
#include "stdafx.h"

struct IDirectSound8;
struct IDirectSoundBuffer;

namespace tbd
{
    enum SoundType
    {
        SOUND_FX,
        SPEECH
    };

    class ResHandle;

    class ISoundBuffer
    {
    public:
        virtual BOOL VInit(VOID) = 0;

        virtual VOID VPlay(BOOL loop = FALSE) = 0;

        virtual VOID VStop(VOID) = 0;

        virtual VOID VPause(VOID) = 0;

        virtual VOID VSetVolume(LONG volume) = 0;

        virtual VOID VResume(VOID) = 0;

        virtual INT VGetVolume(VOID) = 0;

        virtual BOOL VIsPlaying(VOID) = 0;

        virtual BOOL VRestore(VOID) = 0;

        virtual VOID VSetPan(FLOAT pan) = 0; //-1 left 0 middle 1 right

        virtual BOOL VIsLooping(VOID) = 0;

        virtual FLOAT VGetProgress(VOID) = 0;

        virtual ~ISoundBuffer(VOID) {}
    };

    class ISoundSystem
    {
    public:
        virtual BOOL VInit(VOID) = 0;

        virtual ISoundBuffer* VCreateSoundBuffer(std::shared_ptr<tbd::ResHandle> handle) = 0;

        virtual VOID VReleaseSoundBuffer(ISoundBuffer* buffer) = 0;

        virtual VOID VStopAll(VOID) = 0;

        virtual VOID VResumeAll(VOID) = 0;

        virtual VOID VPauseAll(VOID) = 0;

        virtual ~ISoundSystem(VOID) {}
    };

    class BaseSoundSystem : public ISoundSystem
    {
    protected:
        std::list<ISoundBuffer*> m_currentSoundBuffer;

    public:
        virtual VOID VStopAll(VOID);

        virtual VOID VResumeAll(VOID);

        virtual VOID VPauseAll(VOID);

        virtual VOID VReleaseSoundBuffer(ISoundBuffer* buffer) {}

        virtual ~BaseSoundSystem(VOID);
    };

    class DirectSoundSystem : public BaseSoundSystem
    {
        friend class DirectWaveBuffer;

        IDirectSound8* m_pDirectSound;
        IDirectSoundBuffer* m_pPrimaryBuffer;
        BOOL m_initialized;
        BOOL m_loop;
    public:
        DirectSoundSystem(VOID);

        BOOL VInit(VOID);

        ISoundBuffer* VCreateSoundBuffer(std::shared_ptr<tbd::ResHandle> handle);

        VOID VReleaseSoundBuffer(ISoundBuffer* buffer);

        ~DirectSoundSystem(VOID);
    };

    class DirectWaveBuffer : public ISoundBuffer
    {
    private:
        DirectSoundSystem* m_pSystem;
        std::shared_ptr<tbd::ResHandle> m_pHandle;
        BOOL m_initialized;
        IDirectSoundBuffer* m_pSoundBuffer;
        LPDWORD m_position;

        BOOL FillBuffer(VOID);

    public:
        DirectWaveBuffer(DirectSoundSystem* system, std::shared_ptr<tbd::ResHandle> handle);
        
        BOOL VInit(VOID);

        VOID VPlay(BOOL looping = FALSE);

        VOID VStop(VOID);

        VOID VPause(VOID);

        VOID VSetVolume(LONG volume);

        VOID VResume(VOID);

        INT VGetVolume(VOID);

        VOID VSetPan(FLOAT pan); //-1 left 0 middle 1 right

        BOOL VIsPlaying(VOID);

        BOOL VRestore(VOID);

        BOOL VIsLooping(VOID);

        FLOAT VGetProgress(VOID);

        ~DirectWaveBuffer(VOID);
    };
}
