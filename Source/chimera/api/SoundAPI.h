#pragma once
#include "CMTypes.h"

namespace chimera
{
    class ISoundBuffer
    {
    public:
        virtual bool VInit(void) = 0;

        virtual void VPlay(bool loop = false) = 0;

        virtual void VStop(void) = 0;

        virtual void VPause(void) = 0;

        virtual void VSetVolume(LONG volume) = 0;

        virtual void VResume(void) = 0;

        virtual int VGetVolume(void) = 0;

        virtual bool VIsPlaying(void) = 0;

        virtual bool VRestore(void) = 0;

        virtual void VSetPan(float pan) = 0; //-1 left 0 middle 1 right

        virtual bool VIsLooping(void) = 0;

        virtual float VGetProgress(void) = 0;

        virtual ~ISoundBuffer(void) {}
    };

    class ISoundEngine
    {

    };

    class ISoundSystem
    {
    public:
        virtual bool VInit(void) = 0;

        virtual ISoundBuffer* VCreateSoundBuffer(std::shared_ptr<IResHandle> handle) = 0;

        virtual void VReleaseSoundBuffer(ISoundBuffer* buffer) = 0;

        virtual void VStopAll(void) = 0;

        virtual void VResumeAll(void) = 0;

        virtual void VPauseAll(void) = 0;

        virtual ~ISoundSystem(void) {}
    };

    class ISoundFactory
    {
    public:
        virtual ISoundSystem* VCreateSoundSystem(void) = 0;
    };
}
