#pragma once
#include "stdafx.h"
#include "GraphicsSettings.h"
#include "Input.h"
#include "Event.h"
#include "Timer.h"
#include "SoundEngine.h"

typedef UINT GameViewId;

namespace d3d
{
    class D3DRenderer;
}
namespace tbd 
{
    namespace gui
    {
        class D3D_GUI;
        class GuiConsole;
    }
    class IScreenElement;
    class ISoundSystem;
    class SceneGraph;
    class VRamManager;
    class IGraphicsSettings;
    class IPicker;
    class ParticleManager;
    class Actor;
    class CameraComponent;

    enum GameViewType 
    {
        HUMAN,
        CONTROLLER,
        AI,
        OTHER
    };

    class IGameView
    {
    protected:
        GameViewId m_id;
        //ActorId m_actorId;
        std::shared_ptr<tbd::Actor> m_actor;
        std::string m_name;
    public:
        IGameView(VOID);

        GameViewId GetId(VOID) { return m_id; }
        std::shared_ptr<tbd::Actor> IGameView::GetTarget(VOID);
        CONST std::string& GetName(VOID) CONST;
        VOID SetName(CONST std::string& name);

        virtual HRESULT VOnRestore(VOID) { return S_OK; };
        virtual VOID VOnRender(DOUBLE time, FLOAT elapsedTime) {}
        virtual VOID VPreRender(VOID) {}
        virtual VOID VPostRender(VOID) {}
        virtual VOID VOnUpdate(ULONG deltaMillis) = 0;
        virtual VOID VOnLosteDevice(VOID) { };

        virtual VOID VOnAttach(UINT viewId, std::shared_ptr<tbd::Actor> actor);

        virtual VOID VSetTarget(std::shared_ptr<tbd::Actor> actor);

        virtual GameViewType VGetType(VOID) CONST = 0;

        virtual ~IGameView(VOID) {}
    };

    class HumanGameView : public IGameView 
    {
    private:
        tbd::SceneGraph* m_pSceneGraph;
        //tbd::IGraphicsSettings* m_pGraphisSettings[3];
        BYTE m_loadingDots;
        tbd::IPicker* m_picker;
        tbd::ParticleManager* m_pParticleManager;
        tbd::gui::D3D_GUI* m_pGui;
        tbd::SoundEngine m_soundEngine;
        tbd::ISoundSystem* m_pSoundSystem;
        std::vector<tbd::IScreenElement*> m_screenElements;
        std::vector<tbd::RenderScreen*> m_scenes;
        tbd::RenderScreen* m_currentScene;

    public:
        HumanGameView(VOID);

        HRESULT VOnRestore(VOID);

        VOID VOnRender(DOUBLE time, FLOAT elapsedTime);
    
        VOID VPostRender(VOID);

        d3d::D3DRenderer* GetRenderer(VOID);

        VOID VOnUpdate(ULONG deltaMillis);
    
        VOID VOnAttach(GameViewId viewId, std::shared_ptr<tbd::Actor> actor);

        VOID VOnLosteDevice(VOID) { /*TODO*/ }

        VOID VSetTarget(std::shared_ptr<tbd::Actor> actor);

        tbd::IPicker* GetPicker(VOID) { return m_picker; }

        //VOID SetGraphicsSettings(GraphicsSettingType type) { m_graphicsSettingsType = type; }

        //GraphicsSettingType GetGraphicsSettingsType(VOID) { return m_graphicsSettingsType; }

        //IGraphicsSettings* GetGraphicsSettings(VOID) { return m_pGraphisSettings[m_graphicsSettingsType]; }

        tbd::SoundEngine* GetSoundEngine(VOID) { return &m_soundEngine; }

        tbd::ISoundSystem* GetSoundSystem(VOID) { return m_pSoundSystem; }

        GameViewType VGetType(VOID) CONST { return HUMAN; }

        tbd::SceneGraph* GetSceneGraph(VOID) { return this->m_pSceneGraph; }

        tbd::VRamManager* GetVRamManager(VOID) CONST;

        tbd::ParticleManager* GetParticleManager(VOID) CONST { return m_pParticleManager; }

        std::shared_ptr<tbd::Actor> GetTarget(VOID) CONST { return m_actor; }

        VOID AddScreenElement(tbd::IScreenElement* element);

        VOID AddScene(tbd::RenderScreen* screen);

        VOID ActivateScene(LPCSTR name);

        tbd::RenderScreen* GetSceneByName(LPCSTR name);

        tbd::IScreenElement* GetScreenElementByName(LPCSTR name); //this is slowly implemented

        VOID ToggleConsole(VOID);

        tbd::gui::D3D_GUI* GetGUI(VOID) CONST { return m_pGui; }

        tbd::gui::GuiConsole* GetConsole(VOID);

        VOID ActorMovedDelegate(event::IEventPtr eventData);

        VOID NewComponentDelegate(event::IEventPtr pEventData);

        VOID DeleteActorDelegate(event::IEventPtr pEventData);

        VOID LoadingLevelDelegate(event::IEventPtr pEventData);

        VOID LevelLoadedDelegate(event::IEventPtr pEventData);

        VOID SetParentDelegate(event::IEventPtr pEventData);

        VOID Resize(UINT w, UINT h, BOOL fullscreen);

        virtual ~HumanGameView(VOID);
    };

    //todo: move everything below to somewhere else
    typedef fastdelegate::FastDelegate0<VOID> UpdateAction;
        typedef fastdelegate::FastDelegate3<INT, INT, INT> MouseScrollAction;
    class ActorController : public IGameView, public tbd::IKeyListener, public tbd::IMouseListener 
    {
    protected:
        INT m_lastPosX;
        INT m_lastPosY;
        FLOAT m_minSpeed;
        FLOAT m_maxSpeed;

        //todo: do for released
        /*std::map<UINT, KeyboardButtonPressedAction> m_keyBoardButtonPressedActions;
        std::map<UINT, KeyboardButtonPressedAction> m_keyBoardButtonDownActions;
        std::map<UINT, MouseButtonPressedAction> m_mouseButtonPressedActions; */

        std::map<UINT, std::string> m_keyBoadButtonPressedCommand;
        std::map<UINT, std::string> m_keyBoadButtonReleasedCommand;
        std::map<UINT, std::string> m_keyBoadButtonDownCommand;

        MouseScrollAction m_scrollAction;
        UpdateAction m_updateAction;

        UINT m_forwardKey;
        UINT m_backKey;
        UINT m_leftKey;
        UINT m_rightKey;

    public:
        ActorController(VOID);

        VOID SetMinSpeed(FLOAT minSpeed) { m_minSpeed = minSpeed; }
        VOID SetMaxSpeed(FLOAT maxSpeed) { m_maxSpeed = maxSpeed; }

        virtual HRESULT VOnRestore(VOID) { return S_OK; }

        virtual VOID VOnRender(DOUBLE time, FLOAT elapsedTime) { }

        virtual VOID VOnUpdate(ULONG deltaMillis);

        virtual VOID VOnLosteDevice(VOID) { }

        virtual GameViewType VGetType(VOID) CONST { return CONTROLLER; }

        virtual BOOL VOnKeyDown(UINT CONST code);
        virtual BOOL VOnKeyPressed(UINT CONST code);
        virtual BOOL VOnKeyReleased(UINT CONST code);
        virtual BOOL VOnKeyRepeat(UINT CONST code);

        virtual BOOL VOnMouseButtonDown(INT x, INT y, INT button);

        virtual BOOL VOnMouseButtonReleased(INT x, INT y, INT button);
        virtual BOOL VOnMouseButtonPressed(INT x, INT y, INT button);
        virtual BOOL VOnMousePressed(INT x, INT y, INT button);
        virtual BOOL VOnMouseMoved(INT x, INT y, INT dx, INT dy);
        virtual BOOL VOnMouseDragged(INT x, INT y, INT dx, INT dy, INT button);
        virtual BOOL VOnMouseWheel(INT x, INT y, INT delta);

        VOID RegisterKeyPressedCommand(UINT key, CONST std::string& command);
        VOID RegisterKeyReleasedCommand(UINT key, CONST std::string& command);
        VOID RegisterKeyDownCommand(UINT key, CONST std::string& command);

        VOID RegisterKeyCommand(UINT key, CONST std::string& command);
        //todo: do for released
        /*VOID RegisterKeyPressedAction(UINT key, KeyboardButtonPressedAction action);
        VOID RegisterKeyDownAction(UINT key, KeyboardButtonPressedAction action);
        VOID RegisterMousePressedAction(UINT mouseButton, MouseButtonPressedAction action); */
        VOID SetMouseScrollAction(MouseScrollAction action);
        VOID SetUpdateAction(UpdateAction action);

        virtual ~ActorController(VOID) { }
    };

    class CharacterController : public ActorController
    {
    private:
        std::shared_ptr<tbd::CameraComponent> m_cameraComp;
        /*
        BOOL m_editMode;
        BOOL m_bMovePicked;
        FLOAT m_actorPlaceScale;
        ActorId m_toModify;
        BOOL m_kinematicPhysical; */
    public:
        CharacterController(VOID);

        BOOL VOnMouseButtonPressed(INT x, INT y, INT button);
        BOOL VOnMouseMoved(INT x, INT y, INT dx, INT dy);
        VOID VOnUpdate(ULONG millis);
        BOOL VOnKeyDown(UINT CONST code);
        VOID VOnAttach(GameViewId viewId, std::shared_ptr<tbd::Actor> actor);
        VOID VSetTarget(std::shared_ptr<tbd::Actor> actor);
    };
};

