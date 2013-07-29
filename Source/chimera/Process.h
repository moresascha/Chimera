#pragma once
#include "stdafx.h"
#include "Vec3.h"
#include "Event.h"
#include "Locker.h"

namespace view 
{
    class Mesh;
}

namespace tbd
{
    class Actor;
    class ISoundBuffer;
    class TransformComponent;
    class SoundEmitterComponent;
    class LightComponent;
    class ResHandle;
}

namespace cudah
{
    class cudah;
}

namespace util
{
    class ICamera;
}

namespace d3d
{
    class Shader;
}

namespace proc 
{

    enum Type
    {
        NORMAL,
        REALTIME,
        ACTOR_REALTIME
    };

    enum State {
        UNINITIALIZED,
        REMOVED,
        RUNNING,
        PAUSED,
        SUCCEEDED,
        FAILED,
        ABORTED
    };

    class Process
    {
        friend class ProcessManager;
    private:
        State m_state;
    protected:
        std::shared_ptr<Process> m_child;
    protected:
        virtual VOID VOnInit(VOID) {};
        virtual VOID VOnAbort(VOID) {};
        virtual VOID VOnFail(VOID) {};
        virtual VOID VOnSuccess(VOID) {};
        virtual VOID VOnUpdate(ULONG deltaMillis) = 0;
        virtual Type VGetType(VOID) { return NORMAL; }

        VOID SetState(State state) { m_state = state; }

    public:
        Process(VOID) : m_state(UNINITIALIZED) {}

        inline std::shared_ptr<Process> PeekChild(VOID) CONST { return m_child; }
        std::shared_ptr<Process> RemoveChild(VOID);
        virtual VOID VSetChild(std::shared_ptr<Process> child) { this->m_child = child; }

        inline State GetState(VOID) CONST { return m_state; }

        inline VOID Succeed(VOID) { m_state = SUCCEEDED; }
        inline VOID Fail(VOID) { m_state = FAILED; }
        inline VOID Pause(VOID) { m_state = PAUSED; }
        inline VOID UnPause(VOID) { m_state = RUNNING; }

        virtual BOOL IsAlive(VOID) CONST { return m_state == RUNNING || m_state == PAUSED; }
        virtual BOOL IsDead(VOID) CONST { return m_state == SUCCEEDED || m_state == ABORTED || m_state == FAILED; }

        virtual ~Process(VOID) {}
    };

    class RealtimeProcess : public Process 
    {
        friend class ProcessManager;
    private:
        VOID WaitComplete(VOID);

    protected:
        HANDLE m_pThreadHandle;
        HANDLE m_pWaitHandle;
        DWORD m_threadId;
        int m_priority;

        virtual VOID VOnInit(VOID);
        virtual VOID VOnUpdate(ULONG deltaMillis) {}
        virtual VOID VThreadProc(VOID) = 0;

        virtual VOID VOnAbort(VOID);
        virtual VOID VOnSuccess(VOID);
        virtual VOID VOnFail(VOID);

        virtual Type VGetType(VOID) { return REALTIME; }
        VOID SetPriority(int priority) { m_priority = priority; }

    public:
        RealtimeProcess(int priority = THREAD_PRIORITY_NORMAL);
        virtual ~RealtimeProcess(VOID);
        static DWORD WINAPI ThreadProc(LPVOID lpParam);
    };

    class ActorProcess : public Process
    {
    private:
        VOID ActorCreatedDelegate(event::IEventPtr data);
        VOID DeleteActorDelegate(event::IEventPtr data);
        
    protected:
        std::shared_ptr<tbd::Actor> m_actor;
        BOOL m_isCreated;
    public:
        ActorProcess(std::shared_ptr<tbd::Actor> actor);
        virtual VOID VOnInit(VOID);
        virtual VOID VOnActorCreate(VOID) { };
        virtual VOID VOnActorDelete(VOID) { };
        virtual ~ActorProcess(VOID);
    };

    class ActorRealtimeProcess : public RealtimeProcess
    {
    protected:
        std::shared_ptr<tbd::Actor> m_actor;
        HANDLE m_event;
        VOID VThreadProc(VOID);
        Type VGetType(VOID) { return ACTOR_REALTIME; }
    public:
        ActorRealtimeProcess(std::shared_ptr<tbd::Actor> actor);
        VOID ActorCreatedDelegate(event::IEventPtr data);
        VOID DeleteActorDelegate(event::IEventPtr data);
        virtual VOID _VThreadProc(VOID) = 0;
        virtual ~ActorRealtimeProcess(VOID);
    };

    class RotationProcess : public ActorRealtimeProcess
    {
    private:
        util::Vec3 m_rotations;
    public:
        RotationProcess(std::shared_ptr<tbd::Actor> actor, CONST util::Vec3& rotations) : ActorRealtimeProcess(actor), m_rotations(rotations)
        {

        }
        VOID _VThreadProc(VOID);
    };

    class StrobeLightProcess : public RealtimeProcess
    {
    private:
        std::shared_ptr<tbd::LightComponent> m_lightComponent;
        FLOAT m_prob;
        UINT m_freq;
    protected:
        //VOID VOnUpdate(ULONG deltaMillis);
        VOID VThreadProc(VOID);
    public:
        StrobeLightProcess(std::shared_ptr<tbd::LightComponent> lightComponent, FLOAT prob, UINT freq /*millis*/);
    };

    class WatchDirModifacationProcess : public proc::RealtimeProcess
    {
    private:
        VOID Close();
    protected:
        std::wstring m_dir;
        HANDLE m_fileHandle;
        HANDLE m_closeHandle;
        VOID VThreadProc(VOID);

        virtual VOID VOnInit(VOID);
        
        virtual VOID VOnAbort(VOID);
        virtual VOID VOnFail(VOID);
        virtual VOID VOnSuccess(VOID);
    public:
        virtual VOID VOnDirModification(VOID) = 0;

    public:
        WatchDirModifacationProcess(LPCTSTR dir);
        virtual ~WatchDirModifacationProcess(VOID);
    };

    class WatchFileModificationProcess : public WatchDirModifacationProcess
    {
    protected:
        BOOL m_update;
        std::wstring m_file;
        time_t m_lastModification;
        virtual VOID VOnInit(VOID);
        time_t GetTime(VOID);
        VOID VOnUpdate(ULONG deltaMillis);
        VOID VOnFail(VOID);
    public:
        WatchFileModificationProcess(LPCTSTR file, LPCTSTR dir);
        VOID VOnDirModification(VOID);
        virtual VOID VOnFileModification(VOID) = 0;
    };

    class WatchShaderFileModificationProcess : public WatchFileModificationProcess
    {
    private:
        d3d::Shader* m_shader;
    public:
        WatchShaderFileModificationProcess(d3d::Shader* shader, LPCTSTR file, LPCTSTR dir);
        VOID VOnFileModification(VOID);
    };

    class WatchCudaFileModificationProcess : public WatchFileModificationProcess
    {
    private:
        cudah::cudah* m_pCuda;
    public:
        WatchCudaFileModificationProcess(cudah::cudah* cuda, LPCTSTR file, LPCTSTR dir);
        VOID VOnFileModification(VOID);
    };

    class SoundProcess : public ActorProcess
    {
    protected:
        std::shared_ptr<tbd::ResHandle> m_pHandle;
        tbd::ISoundBuffer* m_pSoundBuffer;
        INT m_soundType;
        INT m_volume;
        INT m_pan;
        BOOL m_loop;

        virtual VOID VOnInit(VOID);
        virtual VOID VOnAbort(VOID);
        virtual VOID VOnFail(VOID);
        virtual VOID VOnSuccess(VOID);
        virtual VOID VOnUpdate(ULONG deltaMillis);

        VOID ComputeVolumeFromDistance(CONST util::Vec3& soundPosition, util::ICamera* camera, FLOAT radius);
    public:
        SoundProcess(std::shared_ptr<tbd::ResHandle> handle, INT soundType = 0/*SOUND_FX*/, INT volume = 100, BOOL loop = FALSE);
        tbd::ISoundBuffer* GetSoundBuffer(VOID) { return m_pSoundBuffer; }
        virtual ~SoundProcess(VOID) {}
    };

    class SoundEmitterProcess : public SoundProcess
    {
    private:
        std::shared_ptr<tbd::TransformComponent> m_transform;
        FLOAT m_radius;
    protected:
        VOID VOnUpdate(ULONG deltaMillis);

    public:
        SoundEmitterProcess(
            std::shared_ptr<tbd::Actor> actor, 
            std::shared_ptr<tbd::TransformComponent> transCmp, 
            std::shared_ptr<tbd::ResHandle> handle,
            FLOAT radius,
            INT soundType = 0/*SOUND_FX*/, INT volume = 100, BOOL loop = FALSE);

        VOID VOnActorDelete(VOID);
    };

    class StaticSoundEmitterProcess : public SoundProcess
    {
    private:
        util::Vec3 m_position;
        FLOAT m_radius;
    public:
        StaticSoundEmitterProcess(
            CONST util::Vec3& position,
            std::shared_ptr<tbd::ResHandle> handle,
            FLOAT radius,
            INT soundType = 0/*SOUND_FX*/, INT volume = 100, BOOL loop = FALSE);
        VOID VOnUpdate(ULONG deltaMillis);
        VOID VOnInit(VOID);
    };
};

