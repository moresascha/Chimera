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
        IPicker* m_pActorPicker;
        bool m_isFullscreen;
        IProcessManager* m_pProcessManager;

    public:
        HumanGameView(void);

        bool VOnRestore(void);

        void VOnRender(void);
    
        void VPostRender(void);

        void VSetFullscreen(bool fullscreen);

        bool VIsFullscreen(void);
        
        bool VInitialise(FactoryPtr* facts);

        IRenderer* VGetRenderer(void);

        void VOnUpdate(ulong deltaMillis);
    
        void VOnAttach(ViewId viewId, IActor* actor);

        void VSetTarget(IActor* actor);

        IPicker* VGetPicker(void) { return m_picker; }

        void VAddSceneNodeCreator(SceneNodeCreator nc, ComponentId cmpid);

        void VRemoveSceneNodeCreator(ComponentId cmpid);

        //VOID SetGraphicsSettings(GraphicsSettingType type) { m_graphicsSettingsType = type; }

        //GraphicsSettingType GetGraphicsSettingsType(VOID) { return m_graphicsSettingsType; }

        //IGraphicsSettings* GetGraphicsSettings(VOID) { return m_pGraphisSettings[m_graphicsSettingsType]; }

        IGraphicsFactory* VGetGraphicsFactory(void) { return m_pGraphicsFactory; }

        IGuiFactory* VGetGuiFactory(void) { return m_pGuiFactroy; }

        IVRamManager* VGetVRamManager(void) { return m_pVramManager; }

        IFontManager* VGetFontManager(void) { return m_pFontManager; }

        IEffectFactory* VGetEffectFactory(void) { return m_pEffectFactory; }

        ISoundEngine* VGetSoundEngine(void) { return m_pSoundEngine; }

        ISoundSystem* VGetSoundSystem(void) { return m_pSoundSystem; }

        ViewType VGetType(void) const { return eViewType_Human; }

        ISceneGraph* VGetSceneGraph(void) { return this->m_pSceneGraph; }

        IActor* VGetTarget(void) const { return m_actor; }

        void VAddScreenElement(std::unique_ptr<IScreenElement> element);

        void VAddScene(std::unique_ptr<IRenderScreen> screen);

        void VActivateScene(LPCSTR name);

        IRenderScreen* VGetSceneByName(LPCSTR name);

        IScreenElement* VGetScreenElementByName(LPCSTR name); //this is slowly implemented

        void ToggleConsole(void);

        /*chimera::gui::D3D_GUI* VGetGUI(VOID) CONST { return m_pGui; }

        chimera::gui::GuiConsole* GetConsole(VOID);*/

        void ActorMovedDelegate(IEventPtr eventData);

        void NewComponentDelegate(IEventPtr pEventData);

        void DeleteActorDelegate(IEventPtr pEventData);

        void DeleteComponentDelegate(IEventPtr pEventData);

        void LoadingLevelDelegate(IEventPtr pEventData);

        void LevelLoadedDelegate(IEventPtr pEventData);

        void SetParentDelegate(IEventPtr pEventData);

        void ReleaseChildDelegate(IEventPtr pEventData);

        void VOnResize(uint w, uint h);

        virtual ~HumanGameView(void);
    };

    class ActorController : public IActorController 
    {
    protected:
        int m_lastPosX;
        int m_lastPosY;
        float m_minSpeed;
        float m_maxSpeed;

        //todo: do for released
        /*std::map<UINT, KeyboardButtonPressedAction> m_keyBoardButtonPressedActions;
        std::map<UINT, KeyboardButtonPressedAction> m_keyBoardButtonDownActions;
        std::map<UINT, MouseButtonPressedAction> m_mouseButtonPressedActions; */

        std::map<uint, std::string> m_keyBoadButtonPressedCommand;
        std::map<uint, std::string> m_keyBoadButtonReleasedCommand;
        std::map<uint, std::string> m_keyBoadButtonDownCommand;

        MouseScrollAction m_scrollAction;
        UpdateAction m_updateAction;

        uint m_forwardKey;
        uint m_backKey;
        uint m_leftKey;
        uint m_rightKey;

    public:
        ActorController(void);

        virtual bool VInitialise(FactoryPtr* facts) { return true; }

        void VSetMinSpeed(float minSpeed) { m_minSpeed = minSpeed; }
        void VSetMaxSpeed(float maxSpeed) { m_maxSpeed = maxSpeed; }

        void VActivate(void);

        void VDeactivate(void);

        virtual bool VOnRestore(void) { return true; }

        virtual void VOnRender(DOUBLE time, float elapsedTime) { }

        virtual void VOnUpdate(ulong deltaMillis);

        virtual ViewType VGetType(void) const { return eProjectionType_Controller; }

        virtual bool VOnKeyDown(uint const code);
        virtual bool VOnKeyPressed(uint const code);
        virtual bool VOnKeyReleased(uint const code);
        virtual bool VOnKeyRepeat(uint const code);

        virtual bool VOnMouseButtonDown(int x, int y, int button);

        virtual bool VOnMouseButtonReleased(int x, int y, int button);
        virtual bool VOnMouseButtonPressed(int x, int y, int button);
        virtual bool VOnMousePressed(int x, int y, int button);
        virtual bool VOnMouseMoved(int x, int y, int dx, int dy);
        virtual bool VOnMouseDragged(int x, int y, int dx, int dy, int button);
        virtual bool VOnMouseWheel(int x, int y, int delta);

        void VRegisterKeyPressedCommand(uint key, const std::string& command);
        void VRegisterKeyReleasedCommand(uint key, const std::string& command);
        void VRegisterKeyDownCommand(uint key, const std::string& command);

        void VRegisterKeyCommand(uint key, const std::string& command);

        void VSetMouseScrollAction(MouseScrollAction action);
        void VSetUpdateAction(UpdateAction action);

        virtual ~ActorController(void) { VDeactivate(); }
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
        CharacterController(void);

        bool VOnMouseButtonPressed(int x, int y, int button);
        bool VOnMouseMoved(int x, int y, int dx, int dy);
        bool VOnMouseDragged(int x, int y, int dx, int dy, int button);
        void VOnUpdate(ulong millis);
        bool VOnKeyDown(uint const code);
        void VOnAttach(ViewId viewId, IActor* actor);
        void VSetTarget(IActor* actor);
    };
};

