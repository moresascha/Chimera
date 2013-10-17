#pragma once
#include "CMTypes.h"

namespace chimera
{
    class ILogic
    {
    public:
        ILogic(VOID) {}

        virtual BOOL VInitialise(FactoryPtr* facts) = 0;

        virtual VOID VOnUpdate(ULONG millis) = 0;

        virtual IActor* VCreateActor(LPCSTR resource, BOOL appendToLevel = FALSE) = 0;

        virtual IActor* VCreateActor(std::unique_ptr<ActorDescription> desc, BOOL appendToLevel = FALSE) = 0;

        virtual VOID VAttachView(std::unique_ptr<IView> view, ActorId actor) = 0;

        virtual VOID VRemoveActor(ActorId id) = 0;

        virtual IActor* VFindActor(ActorId id) = 0;

        virtual IActor* VFindActor(LPCSTR name) = 0;

        virtual IView* VFindView(GameViewId id) = 0;

        virtual IView* VFindView(LPCSTR name) = 0;

        virtual ILevel* VGetlevel(VOID) = 0;

        virtual BOOL VLoadLevel(ILevel* level) = 0;

        virtual BOOL VLoadLevel(CONST CHAR* ressource) = 0;

        virtual ICommandInterpreter* VGetCommandInterpreter(VOID) = 0;

        virtual IProcessManager* VGetProcessManager(VOID) = 0;

        virtual IPhysicsSystem* VGetPhysics(VOID) = 0;

        virtual IHumanView* VGetHumanView(VOID) = 0;

        virtual GameState VGetGameState(VOID) = 0;

        virtual IActorFactory* VGetActorFactory(VOID) = 0;

        virtual VOID VSetGameState(GameState state) = 0;

        virtual ~ILogic(VOID) {}
    };

    class ILevel
    {
    public:
        virtual BOOL VLoad(BOOL block) = 0;

        virtual VOID VUnload(VOID) = 0;

        virtual CONST std::string& VGetName(VOID) = 0;

        virtual CONST std::string& VGetFile(VOID) = 0;

        virtual BOOL VSave(LPCSTR file = NULL) = 0;

        virtual UINT VGetActorsCount(VOID) = 0;

        virtual FLOAT VGetLoadingProgress(VOID) = 0;

        virtual IActor* VAddActor(std::unique_ptr<ActorDescription> desc) = 0;

        virtual VOID VRemoveActor(ActorId id) = 0;

        virtual IActor* VFindActor(ActorId id) = 0;

        virtual CONST std::map<ActorId, std::unique_ptr<IActor>>& VGetActors(VOID) = 0;

        virtual ~ILevel(VOID) {}
    };

    class IProcessManager
    {
    public:
        virtual UINT VUpdate(ULONG delatMillis) = 0;

        virtual IProcess* VAttach(std::shared_ptr<IProcess> process) = 0;

        virtual IProcess* VAttachWithScheduler(std::shared_ptr<IProcess> process) = 0;

        virtual  VOID VAbortAll(BOOL immediate) = 0;

        virtual UINT VGetProcessCount(VOID) CONST = 0;
        virtual ~IProcessManager(VOID) {}
    };

    class ICommandInterpreter
    {
    public:
        virtual BOOL VCallCommand(LPCSTR cmd) = 0;
        virtual ~ICommandInterpreter(VOID) {}
    };

    class IScheduler
    {
    public:
        virtual IProcess* VAddProcess(std::shared_ptr<IProcess> proc) = 0;
        virtual ~IScheduler(VOID) {}
    };

    class IProcess
    {
    protected:
        ProcessState m_state;
        std::unique_ptr<IProcess> m_child;
    public:
        virtual VOID VOnInit(VOID) {};
        virtual VOID VOnAbort(VOID) {};
        virtual VOID VOnFail(VOID) {};
        virtual VOID VOnSuccess(VOID) {};
        virtual VOID VOnUpdate(ULONG deltaMillis) = 0;
        virtual ProcessType VGetType(VOID) { return eProcessType_Normal; }

        VOID SetState(ProcessState state) { m_state = state; }

        IProcess(VOID) : m_state(eProcessState_Uninitialized) { }

        CM_INLINE std::unique_ptr<IProcess> RemoveChild(VOID)
        {
            if(m_child)
            {
                return std::move(m_child);
            }

            return NULL;
        }

        virtual VOID VSetChild(std::unique_ptr<IProcess> child) { m_child = std::move(child); }

        CM_INLINE ProcessState GetState(VOID) CONST { return m_state; }

        CM_INLINE VOID Succeed(VOID) { m_state = eProcessState_Succed; }
        CM_INLINE VOID Fail(VOID) { m_state = eProcessState_Failed; }
        CM_INLINE VOID Pause(VOID) { m_state = eProcessState_Paused; }
        CM_INLINE VOID UnPause(VOID) { m_state = eProcessState_Running; }

        virtual BOOL IsAlive(VOID) CONST { return m_state == eProcessState_Running || m_state == eProcessState_Paused; }
        virtual BOOL IsDead(VOID) CONST { return m_state == eProcessState_Succed || m_state == eProcessState_Aborted || m_state == eProcessState_Failed; }

        virtual ~IProcess(VOID) {}
    };

    class ILogicFactory
    {
    public:
        virtual ILogic* VCreateLogic(VOID) = 0;
    };
}