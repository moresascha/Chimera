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

        uint m_levelActorsCount;

        ILevel* m_pLevel;

        IHumanView* m_pHumanView;

        GameState m_gameState;

    public:
        BaseGameLogic(void);

        bool VInitialise(FactoryPtr* facts);

        void VAttachView(std::unique_ptr<IView> view, IActor* actor);

        IActor* VFindActor(ActorId id);

        IActor* VFindActor(LPCSTR name);

        IView* VFindView(ViewId id);

        IView* VFindView(LPCSTR name);

        IHumanView* VGetHumanView(void) { return m_pHumanView; }

        void VOnUpdate(ulong millis);

        IActor* VCreateActor(LPCSTR resource, bool appendToLevel = false);

        IActor* VCreateActor(std::unique_ptr<ActorDescription> desc, bool appendToLevel = false);
   
        void VRemoveActor(ActorId id);

        void VOnRender(void);

        ICommandInterpreter* VGetCommandInterpreter(void) { return m_pCmdInterpreter; }

        GameState VGetGameState(void) { return m_gameState; }

        void VSetGameState(GameState state) { m_gameState = state; }

        bool VLoadLevel(const char* ressource);

        bool VLoadLevel(ILevel* level);

        ILevel* VGetlevel(void) { return m_pLevel;}

        float GetLevelLoadProgress(void) const;

        uint GetLevelActorCount(void) const;

        IPhysicsSystem* VGetPhysics(void) { return m_pPhysics; }

        void MoveActorDelegate(IEventPtr eventData);

        void CreateActorDelegate(IEventPtr eventData);

        void DeleteActorDelegate(IEventPtr eventData);

        void ActorCreatedDelegate(IEventPtr eventData);

        void LoadLevelDelegate(IEventPtr eventData);

        void CreateProcessDelegate(IEventPtr eventData);

        void LevelLoadedDelegate(IEventPtr eventData);

        IActorFactory* VGetActorFactory(void) { return m_pActorFactory; }

        IProcessManager* VGetProcessManager(void) { return m_pProcessManager; }

        virtual ~BaseGameLogic(void);
    };
}