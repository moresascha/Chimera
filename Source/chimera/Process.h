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
        void VOnUpdate(ulong deltaMillis)
        {
            m_pF(deltaMillis);
        }
    };

    typedef void (*function_ptr)(uint); 
    class FunctionProcess : public IProcess
    {
    private:
        function_ptr m_pF;
    public:
        FunctionProcess(function_ptr fp) : m_pF(fp)
        {

        }
        void VOnUpdate(ulong deltaMillis)
        {
            m_pF(deltaMillis);
        }
    };

    typedef fastdelegate::FastDelegate2<ulong, IActor*> _AFDelegate;
    class ActorDelegateProcess : public IProcess
    {
    private:
        _AFDelegate m_pF;
        IActor* m_actor;
    public:
        ActorDelegateProcess(_AFDelegate fp, IActor* actor) : m_pF(fp), m_actor(actor)
        {

        }
        void VOnUpdate(ulong deltaMillis)
        {
            m_pF(deltaMillis, m_actor);
        }
    };

    class RealtimeProcess : public IProcess 
    {
        friend class ProcessManager;
    private:
        void WaitComplete(void);

    protected:
        HANDLE m_pThreadHandle;
        HANDLE m_pWaitHandle;
        DWORD m_threadId;
        int m_priority;

        virtual void VOnInit(void);
        virtual void VOnUpdate(ulong deltaMillis) {}
        virtual void VThreadProc(void) = 0;

        virtual void VOnAbort(void);
        virtual void VOnSuccess(void);
        virtual void VOnFail(void);

        virtual ProcessType VGetType(void) { return eProcessType_Realtime; }
        void SetPriority(int priority) { m_priority = priority; }

    public:
        RealtimeProcess(int priority = THREAD_PRIORITY_NORMAL);
        virtual ~RealtimeProcess(void);
        static DWORD WINAPI ThreadProc(LPVOID lpParam);
    };

    class ActorProcess : public IProcess
    {
    private:
        void ActorCreatedDelegate(IEventPtr data);
        void DeleteActorDelegate(IEventPtr data);
        
    protected:
        IActor* m_actor;
        bool m_isCreated;
    public:
        ActorProcess(IActor* actor);
        virtual void VOnInit(void);
        virtual void VOnActorCreate(void) { };
        virtual void VOnActorDelete(void) { };
        virtual ~ActorProcess(void);
    };

    class ActorRealtimeProcess : public RealtimeProcess
    {
    protected:
        IActor* m_actor;
        HANDLE m_event;
        void VThreadProc(void);
        ProcessType VGetType(void) { return eProcessType_Actor_Realtime; }
    public:
        ActorRealtimeProcess(IActor* actor);
        void ActorCreatedDelegate(IEventPtr data);
        void DeleteActorDelegate(IEventPtr data);
        virtual void _VThreadProc(void) = 0;
        virtual ~ActorRealtimeProcess(void);
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
        void Close();
    protected:
        std::wstring m_dir;
        HANDLE m_fileHandle;
        HANDLE m_closeHandle;
        void VThreadProc(void);

        virtual void VOnInit(void);
        
        virtual void VOnAbort(void);
        virtual void VOnFail(void);
        virtual void VOnSuccess(void);
    public:
        virtual void VOnDirModification(void) = 0;

    public:
        WatchDirModifacationProcess(LPCTSTR dir);
        virtual ~WatchDirModifacationProcess(void);
    };

    class WatchFileModificationProcess : public WatchDirModifacationProcess
    {
    protected:
        bool m_update;
        std::wstring m_file;
        time_t m_lastModification;
        virtual void VOnInit(void);
        time_t GetTime(void);
        void VOnUpdate(ulong deltaMillis);
        void VOnFail(void);
    public:
        WatchFileModificationProcess(LPCTSTR file, LPCTSTR dir);
        void VOnDirModification(void);
        virtual void VOnFileModification(void) = 0;
    };

    class FileWatcherProcess : public WatchFileModificationProcess
    {
    public:
        FileWatcherProcess(LPCTSTR file, LPCTSTR dir);
        FileWatcherProcess(LPCSTR file, LPCSTR dir);
        void VOnFileModification(void);
    };

    class WatchShaderFileModificationProcess : public WatchFileModificationProcess
    {
    private:
        IShader* m_shader;
    public:
        WatchShaderFileModificationProcess(IShader* shader, LPCTSTR file, LPCTSTR dir);
        void VOnFileModification(void);
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
        int m_soundType;
        int m_volume;
        int m_pan;
        bool m_loop;

        virtual void VOnInit(void);
        virtual void VOnAbort(void);
        virtual void VOnFail(void);
        virtual void VOnSuccess(void);
        virtual void VOnUpdate(ulong deltaMillis);

        void ComputeVolumeFromDistance(const util::Vec3& soundPosition, ICamera* camera, float radius);
    public:
        SoundProcess(std::shared_ptr<IResHandle> handle, int soundType = 0/*SOUND_FX*/, int volume = 100, bool loop = false);
        ISoundBuffer* GetSoundBuffer(void) { return m_pSoundBuffer; }
        virtual ~SoundProcess(void) {}
    };

    class SoundEmitterProcess : public SoundProcess
    {
    private:
        TransformComponent* m_transform;
        float m_radius;
    protected:
        void VOnUpdate(ulong deltaMillis);

    public:
        SoundEmitterProcess(
            IActor* actor, 
            TransformComponent* transCmp, 
            std::shared_ptr<IResHandle> handle,
            float radius,
            int soundType = 0/*SOUND_FX*/, int volume = 100, bool loop = false);

        void VOnActorDelete(void);
    };

    class StaticSoundEmitterProcess : public SoundProcess
    {
    private:
        util::Vec3 m_position;
        float m_radius;
    public:
        StaticSoundEmitterProcess(
            const util::Vec3& position,
            std::shared_ptr<IResHandle> handle,
            float radius,
            int soundType = 0/*SOUND_FX*/, int volume = 100, bool loop = false);
        void VOnUpdate(ulong deltaMillis);
        void VOnInit(void);
    };
};

