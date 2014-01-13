#pragma once
#include "stdafx.h"

namespace chimera 
{
    class BaseGameLogic : public ILogic 
    {

    protected:
        IProcessManager* m_pProcessManager;

        IPhysicsSystem* m_pPhysics;
        
        ICommandInterpreter* m_pCmdInterpreter;

        std::vector<std::unique_ptr<IView>> m_gameViewList;

        std::map<ActorId, IView*> m_actorToViewMap;

        std::map<ActorId, std::unique_ptr<IActor>> m_actors;

        IActorFactory* m_pActorFactory;

        std::vector<ActorId> m_levelActors;

        UINT m_levelActorsCount;

        ILevel* m_pLevel;

        IHumanView* m_pHumanView;

        GameState m_gameState;

    public:
        BaseGameLogic(VOID);

        BOOL VInitialise(FactoryPtr* facts);

        VOID VAttachView(std::unique_ptr<IView> view, IActor* actor);

        IActor* VFindActor(ActorId id);

        IActor* VFindActor(LPCSTR name);

        IView* VFindView(ViewId id);

        IView* VFindView(LPCSTR name);

        IHumanView* VGetHumanView(VOID) { return m_pHumanView; }

        VOID VOnUpdate(ULONG millis);

        IActor* VCreateActor(LPCSTR resource, BOOL appendToLevel = FALSE);

        IActor* VCreateActor(std::unique_ptr<ActorDescription> desc, BOOL appendToLevel = FALSE);
   
        VOID VRemoveActor(ActorId id);

        VOID VOnRender(VOID);

        ICommandInterpreter* VGetCommandInterpreter(VOID) { return m_pCmdInterpreter; }

        GameState VGetGameState(VOID) { return m_gameState; }

        VOID VSetGameState(GameState state) { m_gameState = state; }

        BOOL VLoadLevel(CONST CHAR* ressource);

        BOOL VLoadLevel(ILevel* level);

        ILevel* VGetlevel(VOID) { return m_pLevel;}

        FLOAT GetLevelLoadProgress(VOID) CONST;

        UINT GetLevelActorCount(VOID) CONST;

        IPhysicsSystem* VGetPhysics(VOID) { return m_pPhysics; }

        VOID MoveActorDelegate(IEventPtr eventData);

        VOID CreateActorDelegate(IEventPtr eventData);

        VOID DeleteActorDelegate(IEventPtr eventData);

        VOID ActorCreatedDelegate(IEventPtr eventData);

        VOID LoadLevelDelegate(IEventPtr eventData);

        VOID CreateProcessDelegate(IEventPtr eventData);

        VOID LevelLoadedDelegate(IEventPtr eventData);

        IActorFactory* VGetActorFactory(VOID) { return m_pActorFactory; }

        IProcessManager* VGetProcessManager(VOID) { return m_pProcessManager; }

        virtual ~BaseGameLogic(VOID);
    };
}