#pragma once
#include "CMTypes.h"

namespace chimera
{
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

    class ISoundEngine
    {

    };

    class ISoundSystem
    {
    public:
        virtual BOOL VInit(VOID) = 0;

        virtual ISoundBuffer* VCreateSoundBuffer(std::shared_ptr<IResHandle> handle) = 0;

        virtual VOID VReleaseSoundBuffer(ISoundBuffer* buffer) = 0;

        virtual VOID VStopAll(VOID) = 0;

        virtual VOID VResumeAll(VOID) = 0;

        virtual VOID VPauseAll(VOID) = 0;

        virtual ~ISoundSystem(VOID) {}
    };

    class ISoundFactory
    {
    public:
        virtual ISoundSystem* VCreateSoundSystem(VOID) = 0;
    };
}
