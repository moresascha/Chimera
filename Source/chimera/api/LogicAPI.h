#pragma once
#include "CMTypes.h"

namespace chimera
{
    class ILogic
    {
    public:
        ILogic(void) {}

        virtual bool VInitialise(FactoryPtr* facts) = 0;

        virtual void VOnUpdate(ulong millis) = 0;

        virtual IActor* VCreateActor(LPCSTR resource, std::vector<IActor*>* children = NULL, bool appendToLevel = false) = 0;

        virtual IActor* VCreateActor(std::unique_ptr<ActorDescription> desc, bool appendToLevel = false) = 0;

        virtual void VAttachView(std::unique_ptr<IView> view, IActor* actor) = 0;

        virtual void VRemoveActor(ActorId id) = 0;

        virtual IActor* VFindActor(ActorId id) = 0;

        virtual IActor* VFindActor(LPCSTR name) = 0;

        virtual IView* VFindView(ViewId id) = 0;

        virtual IView* VFindView(LPCSTR name) = 0;

        virtual ILevel* VGetlevel(void) = 0;

        virtual bool VLoadLevel(ILevel* level) = 0;

        virtual bool VLoadLevel(const char* ressource) = 0;

        virtual ICommandInterpreter* VGetCommandInterpreter(void) = 0;

        virtual IProcessManager* VGetProcessManager(void) = 0;

        virtual IPhysicsSystem* VGetPhysics(void) = 0;

        virtual IHumanView* VGetHumanView(void) = 0;

        virtual GameState VGetGameState(void) = 0;

        virtual IActorFactory* VGetActorFactory(void) = 0;

        virtual void VSetGameState(GameState state) = 0;

        virtual ~ILogic(void) {}
    };

    class ILevel
    {
    public:
        virtual bool VLoad(bool block) = 0;

        virtual void VUnload(void) = 0;

        virtual const std::string& VGetName(void) = 0;

        virtual const std::string& VGetFile(void) = 0;

        virtual bool VSave(LPCSTR file = NULL) = 0;

        virtual uint VGetActorsCount(void) = 0;

        virtual float VGetLoadingProgress(void) = 0;

        virtual IActor* VAddActor(std::unique_ptr<ActorDescription> desc) = 0;

        virtual void VRemoveActor(ActorId id) = 0;

        virtual IActor* VFindActor(ActorId id) = 0;

        virtual const std::map<ActorId, std::unique_ptr<IActor>>& VGetActors(void) = 0;

        virtual ~ILevel(void) {}
    };

    class IProcessManager
    {
    public:
        virtual uint VUpdate(ulong delatMillis) = 0;

        virtual IProcess* VAttach(std::shared_ptr<IProcess> process) = 0;

        virtual IProcess* VAttachWithScheduler(std::shared_ptr<IProcess> process) = 0;

        virtual  void VAbortAll(bool immediate) = 0;

        virtual uint VGetProcessCount(void) const = 0;
        virtual ~IProcessManager(void) {}
    };

    class ICommand
    {
    public:
        
        virtual bool VInitArgumentTypes(int args, ...) = 0;

        virtual float VGetNextFloat(void) = 0;
        
        virtual int VGetNextInt(void) = 0;
        
        virtual char VGetNextChar(void) = 0;
        
        virtual bool VGetNextBool(void) = 0;
        
        virtual std::string VGetNextCharStr(void) = 0;
        
        virtual std::string VGetRemainingString(void) = 0;

        virtual bool VIsError(void) = 0;
        
        virtual bool VIsValid(void) = 0;

        virtual ~ICommand(void) { }
    };

    class ICommandInterpreter
    {
    public:
        virtual bool VCallCommand(LPCSTR cmd) = 0;

        virtual std::vector<std::string> VGetCommands(void) = 0;

        virtual void VLoadCommands(LPCSTR file) = 0;

        virtual void VRegisterCommand(LPCSTR name, CommandHandler command, LPCSTR usage = NULL) = 0;

        virtual ~ICommandInterpreter(void) {} 
    };

    class IScheduler
    {
    public:
        virtual IProcess* VAddProcess(std::shared_ptr<IProcess> proc) = 0;
        virtual ~IScheduler(void) {}
    };

    class IProcess
    {
    protected:
        ProcessState m_state;
        std::unique_ptr<IProcess> m_child;
    public:
        virtual void VOnInit(void) {};
        virtual void VOnAbort(void) {};
        virtual void VOnFail(void) {};
        virtual void VOnSuccess(void) {};
        virtual void VOnUpdate(ulong deltaMillis) = 0;
        virtual ProcessType VGetType(void) { return eProcessType_Normal; }

        void SetState(ProcessState state) { m_state = state; }

        IProcess(void) : m_state(eProcessState_Uninitialized) { }

        CM_INLINE std::unique_ptr<IProcess> RemoveChild(void)
        {
            if(m_child)
            {
                return std::move(m_child);
            }

            return NULL;
        }

        virtual void VSetChild(std::unique_ptr<IProcess> child) { m_child = std::move(child); }

        CM_INLINE ProcessState GetState(void) const { return m_state; }

        CM_INLINE void Succeed(void) { m_state = eProcessState_Succed; }
        CM_INLINE void Fail(void) { m_state = eProcessState_Failed; }
        CM_INLINE void Pause(void) { m_state = eProcessState_Paused; }
        CM_INLINE void UnPause(void) { m_state = eProcessState_Running; }

        virtual bool IsAlive(void) const { return m_state == eProcessState_Running || m_state == eProcessState_Paused; }
        virtual bool IsDead(void) const { return m_state == eProcessState_Succed || m_state == eProcessState_Aborted || m_state == eProcessState_Failed; }

        virtual ~IProcess(void) {}
    };

    class ILogicFactory
    {
    public:
        virtual ILogic* VCreateLogic(void) = 0;
    };
}