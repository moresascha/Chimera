#include "GameView.h"
#include "Event.h"
#include "Components.h"

namespace chimera
{
    ActorController::ActorController(VOID) 
        : m_lastPosX(-1), m_lastPosY(-1), m_minSpeed(1), m_maxSpeed(3), m_updateAction(NULL), m_scrollAction(NULL)
        , m_leftKey(KEY_A), m_rightKey(KEY_D), m_forwardKey(KEY_W), m_backKey(KEY_S)
    {
    }

    VOID ActorController::VOnUpdate(ULONG millis) 
    {
        if(m_updateAction)
        {
            m_updateAction();
        }
    }

    VOID ActorController::VActivate(VOID)
    {
        CmGetApp()->VGetInputHandler()->VPushKeyListener(this);
        CmGetApp()->VGetInputHandler()->VPushMouseListener(this);
    }

    VOID ActorController::VDeactivate(VOID)
    {
        CmGetApp()->VGetInputHandler()->VRemoveKeyListener(this);
        CmGetApp()->VGetInputHandler()->VRemoveMouseListener(this);
    }

    BOOL ActorController::VOnKeyDown(UINT CONST code) 
    {
        auto itt = m_keyBoadButtonDownCommand.find(code);
        if(itt != m_keyBoadButtonDownCommand.end())
        {
            CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand(itt->second.c_str());
        }

        return TRUE;
    }

    BOOL ActorController::VOnKeyRepeat(UINT CONST code) 
    {
        return TRUE;
    }

    BOOL ActorController::VOnKeyPressed(UINT CONST code) 
    {
        auto itt = m_keyBoadButtonPressedCommand.find(code);
        if(itt != m_keyBoadButtonPressedCommand.end())
        {
            CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand(itt->second.c_str());
        }
        return TRUE;
    }

    BOOL ActorController::VOnKeyReleased(UINT CONST code)
    {
        auto itt = m_keyBoadButtonReleasedCommand.find(code);
        if(itt != m_keyBoadButtonReleasedCommand.end())
        {
            CmGetApp()->VGetLogic()->VGetCommandInterpreter()->VCallCommand(itt->second.c_str());
        }
        return TRUE;
    }

    BOOL ActorController::VOnMouseButtonDown(INT x, INT y, INT button) 
    { 
        this->m_lastPosX = x;
        this->m_lastPosY = y;
        return TRUE; 
    };

    BOOL ActorController::VOnMouseButtonReleased(INT x, INT y, INT button) 
    { 
        return FALSE; 
    }

    BOOL ActorController::VOnMouseButtonPressed(INT x, INT y, INT button) 
    {
        return TRUE; 
    }

    BOOL ActorController::VOnMousePressed(INT x, INT y, INT button) { return FALSE; } //remove?
    BOOL ActorController::VOnMouseMoved(INT x, INT y, INT dx, INT dy) 
    {
        LOG_CRITICAL_ERROR("this does currently not work");
        QUEUE_EVENT(new MoveActorEvent(this->m_actor->GetId(), util::Vec3(2 * dx * 1e-3f, 2 * dy * 1e-3f, 0)));
        this->m_lastPosX = x;
        this->m_lastPosY = y;
        return TRUE;
    }

    BOOL ActorController::VOnMouseDragged(INT x, INT y, INT dx, INT dy, INT button) { return FALSE; }

    BOOL ActorController::VOnMouseWheel(INT x, INT y, INT delta) 
    { 
        if(m_scrollAction)
        {
            m_scrollAction(x, y, delta);
        }
        return TRUE; 
    };

    VOID ActorController::VRegisterKeyPressedCommand(UINT key, CONST std::string& command)
    {
        auto it = m_keyBoadButtonPressedCommand.find(key);

        if(it != m_keyBoadButtonPressedCommand.end())
        {
            //todo
            LOG_CRITICAL_ERROR("overwriting an input action");
        }

        m_keyBoadButtonPressedCommand[key] = command;
    }

    VOID ActorController::VRegisterKeyDownCommand(UINT key, CONST std::string& command)
    {
        auto it = m_keyBoadButtonDownCommand.find(key);

        if(it != m_keyBoadButtonDownCommand.end())
        {
            //todo
            LOG_CRITICAL_ERROR("overwriting an input action");
        }

        m_keyBoadButtonDownCommand[key] = command;
    }

    VOID ActorController::VRegisterKeyReleasedCommand(UINT key, CONST std::string& command)
    {
        auto it = m_keyBoadButtonReleasedCommand.find(key);

        if(it != m_keyBoadButtonReleasedCommand.end())
        {
            //todo
            LOG_CRITICAL_ERROR("overwriting an input action");
        }

        m_keyBoadButtonReleasedCommand[key] = command;
    }

    VOID ActorController::VRegisterKeyCommand(UINT key, CONST std::string& command)
    {
        if(!command.compare("left"))
        {
            m_leftKey = key;
            return;
        }
        if(!command.compare("right"))
        {
            m_rightKey = key;
            return;
        }
        if(!command.compare("forward"))
        {
            m_forwardKey = key;
            return;
        }
        if(!command.compare("back"))
        {
            m_backKey = key;
            return;
        }
        VRegisterKeyPressedCommand(key, command);
    }

    /*
    VOID ActorController::RegisterKeyPressedAction(UINT key, KeyboardButtonPressedAction action)
    {
        auto it = m_keyBoardButtonPressedActions.find(key);

        if(it != m_keyBoardButtonPressedActions.end())
        {
            //todo
            LOG_ERROR("overwriting an input action");
        }

        m_keyBoardButtonPressedActions[key] = action;
    }

    VOID ActorController::RegisterKeyDownAction(UINT key, KeyboardButtonDownAction action)
    {
        auto it = m_keyBoardButtonDownActions.find(key);

        if(it != m_keyBoardButtonDownActions.end())
        {
            //todo
            LOG_ERROR("overwriting an input action");
        }

        m_keyBoardButtonDownActions[key] = action;
    }


    VOID ActorController::RegisterMousePressedAction(UINT mouseButton, MouseButtonPressedAction action)
    {
        auto it = m_mouseButtonPressedActions.find(mouseButton);

        if(it != m_mouseButtonPressedActions.end())
        {
            //todo
            LOG_ERROR("overwriting an input action");
        }

        m_mouseButtonPressedActions[mouseButton] = action;
    } */

    VOID ActorController::VSetMouseScrollAction(MouseScrollAction action)
    {
        m_scrollAction = action;
    }

    VOID ActorController::VSetUpdateAction(UpdateAction action)
    {
        m_updateAction = action;
    }

    CharacterController::CharacterController(VOID)
    {

    }

    BOOL CharacterController::VOnMouseMoved(INT x, INT y, INT dx, INT dy) 
    {
        if(!CmGetApp()->VGetInputHandler()->VIsMouseGrabbed())
        {
            return FALSE;
        }
        
        TransformComponent* comp = GetActorCompnent<TransformComponent>(m_actor, CM_CMP_TRANSFORM);
        comp->m_phi -= -2 * dx * 1e-3f;
        comp->m_theta += 2 * dy * 1e-3f;

        FLOAT t = CLAMP(comp->m_theta, -XM_PIDIV2, XM_PIDIV2);

        util::Mat4 m;
        m.SetRotateX(t);

        util::Mat4 m1;
        m1.SetRotateY(comp->m_phi);

        XMVECTOR q = XMQuaternionMultiply(XMLoadFloat4(&m.GetRotation().m_v), XMLoadFloat4(&m1.GetRotation().m_v));        
        comp->GetTransformation()->SetRotateQuat(q.m128_f32[0], q.m128_f32[1], q.m128_f32[2], q.m128_f32[3]);

        QUEUE_EVENT(new ActorMovedEvent(m_actor));

        return TRUE;
    }

    BOOL CharacterController::VOnKeyDown(UINT CONST code)
    {
        ActorController::VOnKeyDown(code);

        ULONG millis = CmGetApp()->VGetUpdateTimer()->VGetLastMillis();
        //DEBUG_OUT(millis);
        util::Vec3 move;
        FLOAT factor = 1e-3f * millis;
        FLOAT speed = 1;

        if(code == m_forwardKey) 
        {
            move.z = factor;
        }
        if(code == m_backKey) 
        {
            move.z = -factor;
        }
        if(code == m_leftKey) 
        {
            move.x = -factor;
        }
        if(code == m_rightKey) 
        {
            move.x = factor;
        }

        if(move.x != 0 || move.z != 0)
        {
            std::shared_ptr<ICamera> camera = m_cameraComp->GetCamera();

            speed = CmGetApp()->VGetInputHandler()->VIsKeyDown(KEY_LSHIFT) ? m_maxSpeed : m_minSpeed;
            move.Scale(speed);
            util::Vec3 deltaX(camera->GetSideDir());
            deltaX.Scale(move.x);

            util::Vec3 deltaZ(camera->GetViewDirXZ());
            deltaZ.Scale(move.z);

            //util::Vec3 deltaY(0, move.y, 0);
            //deltaY.Scale(move.y);

            //deltaX.Add(deltaY);
            deltaX.Add(deltaZ);

            QUEUE_EVENT(new chimera::MoveActorEvent(this->m_actor->GetId(), deltaX));
        }

        return TRUE;
    }

    VOID CharacterController::VSetTarget(IActor* actor)
    {
        IView::VSetTarget(actor);
        m_cameraComp = GetActorCompnent<CameraComponent>(actor, CM_CMP_CAMERA);
    }

    VOID CharacterController::VOnUpdate(ULONG millis)
    {
        ActorController::VOnUpdate(millis);
    }

    BOOL CharacterController::VOnMouseButtonPressed(INT x, INT y, INT button)
    {
        ActorController::VOnMouseButtonPressed(x, y, button);
        if(!CmGetApp()->VGetInputHandler()->VGrabMouse(button & MOUSE_BTN_RIGHT))
        {
            LOG_CRITICAL_ERROR("failed to grab mouse...");
        }

        return TRUE;
    }

    VOID CharacterController::VOnAttach(ViewId viewId, IActor* actor)
    {
        IView::VOnAttach(viewId, actor);

        if(!actor->VHasComponent(CM_CMP_CAMERA))
        {
            LOG_CRITICAL_ERROR("Actor needs a camera component");
        }
        else
        {
            m_cameraComp = GetActorCompnent<CameraComponent>(actor, CM_CMP_CAMERA);
        }
    }
}
