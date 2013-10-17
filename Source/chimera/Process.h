#pragma once
#include "stdafx.h"
#include "Vec3.h"
#include "Event.h"
#include "Locker.h"
#include "Components.h"

namespace chimera 
{    
    class DelegateProcess : public IProcess
    {
    private:
        _FDelegate m_pF;
    public:
        DelegateProcess(_FDelegate fp) : m_pF(fp)
        {

        }
        VOID VOnUpdate(ULONG deltaMillis)
        {
            m_pF(deltaMillis);
        }
    };

    typedef fastdelegate::FastDelegate2<ULONG, IActor*> _AFDelegate;
    class ActorDelegateProcess : public IProcess
    {
    private:
        _AFDelegate m_pF;
        IActor* m_actor;
    public:
        ActorDelegateProcess(_AFDelegate fp, IActor* actor) : m_pF(fp), m_actor(actor)
        {

        }
        VOID VOnUpdate(ULONG deltaMillis)
        {
            m_pF(deltaMillis, m_actor);
        }
    };

    class RealtimeProcess : public IProcess 
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

        virtual ProcessType VGetType(VOID) { return eProcessType_Realtime; }
        VOID SetPriority(int priority) { m_priority = priority; }

    public:
        RealtimeProcess(int priority = THREAD_PRIORITY_NORMAL);
        virtual ~RealtimeProcess(VOID);
        static DWORD WINAPI ThreadProc(LPVOID lpParam);
    };

    class ActorProcess : public IProcess
    {
    private:
        VOID ActorCreatedDelegate(IEventPtr data);
        VOID DeleteActorDelegate(IEventPtr data);
        
    protected:
        IActor* m_actor;
        BOOL m_isCreated;
    public:
        ActorProcess(IActor* actor);
        virtual VOID VOnInit(VOID);
        virtual VOID VOnActorCreate(VOID) { };
        virtual VOID VOnActorDelete(VOID) { };
        virtual ~ActorProcess(VOID);
    };

    class ActorRealtimeProcess : public RealtimeProcess
    {
    protected:
        IActor* m_actor;
        HANDLE m_event;
        VOID VThreadProc(VOID);
        ProcessType VGetType(VOID) { return eProcessType_Actor_Realtime; }
    public:
        ActorRealtimeProcess(IActor* actor);
        VOID ActorCreatedDelegate(IEventPtr data);
        VOID DeleteActorDelegate(IEventPtr data);
        virtual VOID _VThreadProc(VOID) = 0;
        virtual ~ActorRealtimeProcess(VOID);
    };

    /*class RotationProcess : public ActorRealtimeProcess
    {
    private:
        util::Vec3 m_rotations;
    public:
        RotationProcess(std::shared_ptr<chimera::Actor> actor, CONST util::Vec3& rotations) : ActorRealtimeProcess(actor), m_rotations(rotations)
        {

        }
        VOID _VThreadProc(VOID);
    }; */

    class WatchDirModifacationProcess : public RealtimeProcess
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

    class FileWatcherProcess : public WatchFileModificationProcess
    {
    public:
        FileWatcherProcess(LPCTSTR file, LPCTSTR dir);
        FileWatcherProcess(LPCSTR file, LPCSTR dir);
        VOID VOnFileModification(VOID);
    };

    class WatchShaderFileModificationProcess : public WatchFileModificationProcess
    {
    private:
        IShader* m_shader;
    public:
        WatchShaderFileModificationProcess(IShader* shader, LPCTSTR file, LPCTSTR dir);
        VOID VOnFileModification(VOID);
    };

    /*class WatchCudaFileModificationProcess : public WatchFileModificationProcess
    {
    private:
        cudah::cudah* m_pCuda;
    public:
        WatchCudaFileModificationProcess(cudah::cudah* cuda, LPCTSTR file, LPCTSTR dir);
        VOID VOnFileModification(VOID);
    };*/

    class SoundProcess : public ActorProcess
    {
    protected:
        std::shared_ptr<IResHandle> m_pHandle;
        ISoundBuffer* m_pSoundBuffer;
        INT m_soundType;
        INT m_volume;
        INT m_pan;
        BOOL m_loop;

        virtual VOID VOnInit(VOID);
        virtual VOID VOnAbort(VOID);
        virtual VOID VOnFail(VOID);
        virtual VOID VOnSuccess(VOID);
        virtual VOID VOnUpdate(ULONG deltaMillis);

        VOID ComputeVolumeFromDistance(CONST util::Vec3& soundPosition, ICamera* camera, FLOAT radius);
    public:
        SoundProcess(std::shared_ptr<IResHandle> handle, INT soundType = 0/*SOUND_FX*/, INT volume = 100, BOOL loop = FALSE);
        ISoundBuffer* GetSoundBuffer(VOID) { return m_pSoundBuffer; }
        virtual ~SoundProcess(VOID) {}
    };

    class SoundEmitterProcess : public SoundProcess
    {
    private:
        TransformComponent* m_transform;
        FLOAT m_radius;
    protected:
        VOID VOnUpdate(ULONG deltaMillis);

    public:
        SoundEmitterProcess(
            IActor* actor, 
            TransformComponent* transCmp, 
            std::shared_ptr<IResHandle> handle,
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
            std::shared_ptr<IResHandle> handle,
            FLOAT radius,
            INT soundType = 0/*SOUND_FX*/, INT volume = 100, BOOL loop = FALSE);
        VOID VOnUpdate(ULONG deltaMillis);
        VOID VOnInit(VOID);
    };
};

