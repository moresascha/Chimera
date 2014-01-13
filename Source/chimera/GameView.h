#pragma once
#include "stdafx.h"

namespace chimera 
{
    class HumanGameView : public IHumanView 
    {
    public:
        ISceneGraph* m_pSceneGraph;
        BYTE m_loadingDots;
        IPicker* m_picker;
        //chimera::gui::D3D_GUI* m_pGui;
        ISoundEngine* m_pSoundEngine;
        ISoundSystem* m_pSoundSystem;
        IFontManager* m_pFontManager;
        std::vector<std::unique_ptr<IScreenElement>> m_screenElements;
        std::vector<std::unique_ptr<IRenderScreen>> m_scenes;
        std::map<ViewId, SceneNodeCreator> m_nodeCreators;
        IRenderScreen* m_pCurrentScene;
        std::unique_ptr<IRenderer> m_pRenderer;
        IVRamManager* m_pVramManager;
        IGraphicsFactory* m_pGraphicsFactory;
        IEffectFactory* m_pEffectFactory;
        IGuiFactory* m_pGuiFactroy;
        BOOL m_isFullscreen;

    public:
        HumanGameView(VOID);

        BOOL VOnRestore(VOID);

        VOID VOnRender(VOID);
    
        VOID VPostRender(VOID);

        VOID VSetFullscreen(BOOL fullscreen);

        BOOL VIsFullscreen(VOID);
        
        BOOL VInitialise(FactoryPtr* facts);

        IRenderer* VGetRenderer(VOID);

        VOID VOnUpdate(ULONG deltaMillis);
    
        VOID VOnAttach(ViewId viewId, IActor* actor);

        VOID VSetTarget(IActor* actor);

        IPicker* VGetPicker(VOID) { return m_picker; }

        VOID VAddSceneNodeCreator(SceneNodeCreator nc, ComponentId cmpid);

        VOID VRemoveSceneNodeCreator(ComponentId cmpid);

        //VOID SetGraphicsSettings(GraphicsSettingType type) { m_graphicsSettingsType = type; }

        //GraphicsSettingType GetGraphicsSettingsType(VOID) { return m_graphicsSettingsType; }

        //IGraphicsSettings* GetGraphicsSettings(VOID) { return m_pGraphisSettings[m_graphicsSettingsType]; }

        IGraphicsFactory* VGetGraphicsFactory(VOID) { return m_pGraphicsFactory; }

        IGuiFactory* VGetGuiFactory(VOID) { return m_pGuiFactroy; }

        IVRamManager* VGetVRamManager(VOID) { return m_pVramManager; }

        IFontManager* VGetFontManager(VOID) { return m_pFontManager; }

        IEffectFactory* VGetEffectFactory(VOID) { return m_pEffectFactory; }

        ISoundEngine* VGetSoundEngine(VOID) { return m_pSoundEngine; }

        ISoundSystem* VGetSoundSystem(VOID) { return m_pSoundSystem; }

        ViewType VGetType(VOID) CONST { return eViewType_Human; }

        ISceneGraph* VGetSceneGraph(VOID) { return this->m_pSceneGraph; }

        IActor* VGetTarget(VOID) CONST { return m_actor; }

        VOID VAddScreenElement(std::unique_ptr<IScreenElement> element);

        VOID VAddScene(std::unique_ptr<IRenderScreen> screen);

        VOID VActivateScene(LPCSTR name);

        IRenderScreen* VGetSceneByName(LPCSTR name);

        IScreenElement* VGetScreenElementByName(LPCSTR name); //this is slowly implemented

        VOID ToggleConsole(VOID);

        /*chimera::gui::D3D_GUI* VGetGUI(VOID) CONST { return m_pGui; }

        chimera::gui::GuiConsole* GetConsole(VOID);*/

        VOID ActorMovedDelegate(IEventPtr eventData);

        VOID NewComponentDelegate(IEventPtr pEventData);

        VOID DeleteActorDelegate(IEventPtr pEventData);

        VOID LoadingLevelDelegate(IEventPtr pEventData);

        VOID LevelLoadedDelegate(IEventPtr pEventData);

        VOID SetParentDelegate(IEventPtr pEventData);

        VOID VOnResize(UINT w, UINT h);

        virtual ~HumanGameView(VOID);
    };

    class ActorController : public IActorController 
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

        virtual BOOL VInitialise(FactoryPtr* facts) { return TRUE; }

        VOID VSetMinSpeed(FLOAT minSpeed) { m_minSpeed = minSpeed; }
        VOID VSetMaxSpeed(FLOAT maxSpeed) { m_maxSpeed = maxSpeed; }

        VOID VActivate(VOID);

        VOID VDeactivate(VOID);

        virtual BOOL VOnRestore(VOID) { return TRUE; }

        virtual VOID VOnRender(DOUBLE time, FLOAT elapsedTime) { }

        virtual VOID VOnUpdate(ULONG deltaMillis);

        virtual ViewType VGetType(VOID) CONST { return eProjectionType_Controller; }

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

        VOID VRegisterKeyPressedCommand(UINT key, CONST std::string& command);
        VOID VRegisterKeyReleasedCommand(UINT key, CONST std::string& command);
        VOID VRegisterKeyDownCommand(UINT key, CONST std::string& command);

        VOID VRegisterKeyCommand(UINT key, CONST std::string& command);

        VOID VSetMouseScrollAction(MouseScrollAction action);
        VOID VSetUpdateAction(UpdateAction action);

        virtual ~ActorController(VOID) { VDeactivate(); }
    };

    class CameraComponent;

    class CharacterController : public ActorController
    {
    private:
        CameraComponent* m_cameraComp;
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
        VOID VOnAttach(ViewId viewId, IActor* actor);
        VOID VSetTarget(IActor* actor);
    };
};

