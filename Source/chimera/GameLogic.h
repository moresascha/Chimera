#pragma once
#include "stdafx.h"
#include "ProcessManager.h"
#include "Process.h"
#include "GameView.h"
#include "PhysicsSystem.h"
#include "ActorFactory.h"
#include "Level.h"

namespace tbd
{
    class CommandInterpreter;
}

namespace tbd 
{

    class IGameLogic
    {
    public:
        IGameLogic(VOID) {}
        virtual BOOL VInit(VOID) = 0;
        virtual VOID VOnUpdate(ULONG millis) = 0;
        virtual std::shared_ptr<tbd::Actor> VCreateActor(LPCSTR resource, BOOL appendToLevel = FALSE) = 0;
        virtual std::shared_ptr<tbd::Actor> VCreateActor(tbd::ActorDescription desc, BOOL appendToLevel = FALSE) = 0;
        virtual VOID VRemoveActor(ActorId id) = 0;
        virtual std::shared_ptr<tbd::Actor> VFindActor(ActorId id) = 0;
        virtual std::shared_ptr<tbd::Actor> VFindActor(LPCSTR name) = 0;
        virtual std::shared_ptr<tbd::IGameView> VFindGameView(GameViewId id) = 0;
        virtual std::shared_ptr<tbd::IGameView> VFindGameView(LPCSTR name) = 0;
        virtual ~IGameLogic(VOID) {}
    };

    enum GameState
    {
        eLoadingLevel,
        eRunning,
        ePause
    };

    class BaseGameLogic : public IGameLogic 
    {

    protected:
        proc::ProcessManager* m_pProcessManager;

        tbd::IPhysicsSystem* m_pPhysics;
        
        tbd::CommandInterpreter* m_pCmdInterpreter;

        std::vector<std::shared_ptr<tbd::IGameView>> m_gameViewList;

        std::map<ActorId, std::shared_ptr<tbd::IGameView>> m_actorToViewMap;

        std::map<ActorId, std::shared_ptr<tbd::Actor>> m_actors;

        tbd::ActorFactory m_actorFactory;

        std::vector<ActorId> m_levelActors;

        UINT m_levelActorsCount;

        tbd::ILevel* m_pLevel;

        tbd::LevelManager* m_pLevelManager;

        enum GameState m_gameState;

    public:
        BaseGameLogic(VOID);

        virtual BOOL VInit(VOID);

        VOID AttachGameView(std::shared_ptr<tbd::IGameView> view, std::shared_ptr<tbd::Actor> actor);

        VOID AttachProcess(std::shared_ptr<proc::Process> process);

        std::shared_ptr<tbd::Actor> VFindActor(ActorId id);

        std::shared_ptr<tbd::Actor> VFindActor(LPCSTR name);

        std::shared_ptr<tbd::IGameView> VFindGameView(GameViewId id);

        std::shared_ptr<tbd::IGameView> VFindGameView(LPCSTR name);

        VOID VOnUpdate(ULONG millis);

        std::shared_ptr<tbd::Actor> VCreateActor(CONST CHAR* resource, BOOL appendToLevel = FALSE);

        std::shared_ptr<tbd::Actor> VCreateActor(tbd::ActorDescription desc, BOOL appendToLevel = FALSE);
   
        VOID VRemoveActor(ActorId id);

        VOID VOnRender(VOID);

        tbd::CommandInterpreter* GetCommandInterpreter(VOID) { return m_pCmdInterpreter; }

        tbd::LevelManager* GetLevelManager(VOID) { return m_pLevelManager; }

        GameState GetGameState(VOID) { return m_gameState; }

        VOID SetGameState(GameState state) { m_gameState = state; }

        BOOL VLoadLevel(CONST CHAR* ressource);

        virtual BOOL VLoadLevel(tbd::ILevel* level);

        tbd::ILevel* Getlevel(VOID) { return m_pLevel;}

        FLOAT GetLevelLoadProgress(VOID) CONST;

        UINT GetLevelActorCount(VOID) CONST;

        tbd::IPhysicsSystem* GetPhysics(VOID) { return m_pPhysics; }

        VOID MoveActorDelegate(event::IEventPtr eventData);

        VOID CreateActorDelegate(event::IEventPtr eventData);

        VOID DeleteActorDelegate(event::IEventPtr eventData);

        VOID ActorCreatedDelegate(event::IEventPtr eventData);

        VOID LoadLevelDelegate(event::IEventPtr eventData);

        VOID LevelLoadedDelegate(event::IEventPtr eventData);

        tbd::ActorFactory* GetActorFactory(VOID) { return &m_actorFactory; }

        proc::ProcessManager* GetProcessManager(VOID) { return m_pProcessManager; }

        virtual ~BaseGameLogic(VOID);
    };
}