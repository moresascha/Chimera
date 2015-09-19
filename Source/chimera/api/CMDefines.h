#pragma once
#include <windows.h>

#define CM_API __stdcall

#ifdef __COMPILING_DLL
    #define CM_DLL_API __declspec(dllexport)
#else 
    #define CM_DLL_API __declspec(dllimport)
#endif

#define CM_INLINE __inline

#define CM_INVALID_ACTOR_ID 0

#define CM_CONFIG_FILE "config.ini"
#define CM_CONTROLS_FILE "controls.ini"

#define SHADER_PATH "sShaderPath"

#define DEFERRED_SHADER_NAME "DefShader"
#define DEFERRED_SHADER_FILE L"DefShading.hlsl"
#define DEFERRED_SHADER_VS_FUNCTION "DefShading_VS"
#define DEFERRED_SHADER_FS_FUNCTION "DefShading_PS"
#define DEFERRED_INSTANCED_SHADER_VS_FUNCTION "DefShadingInstanced_VS"

#define DEFERRED_WIREFRAME_SHADER_NAME "DefWireFrameShader"
#define DEFERRED_WIREFRAME_SHADER_FILE L"DefShading.hlsl"
#define DEFERRED_WIREFRAME_SHADER_VS_FUNCTION "DefShading_VS"
#define DEFERRED_WIREFRAME_SHADER_FS_FUNCTION "DefShadingWire_PS"

#define GLOBAL_LIGHTING_SHADER_NAME "GlobalLighting"
#define GLOBAL_LIGHTING_SHADER_FILE L"Lighting.hlsl"
#define GLOBAL_LIGHTING_SHADER_VS_FUNCTION "Lighting_VS"
#define GLOBAL_LIGHTING_SHADER_FS_FUNCTION "GlobalLighting_PS"

#define SCREENQUAD_SHADER_NAME "ScreenQuad"
#define SCREENQUAD_SHADER_FILE L"Effects.hlsl"
#define SCREENQUAD_SHADER_VS_FUNCTION "Effect_VS"
#define SCREENQUAD_SHADER_FS_FUNCTION "SampleDiffuseTexture"

#define VIEW_CONTROLLER_NAME "HumanCameraController"
#define VIEW_CONSOLE_NAME "Console"

#define CM_KEY_TAB 0x09
#define KEY_0 0x30
#define KEY_1 0x31
#define KEY_2 0x32
#define KEY_3 0x33
#define KEY_4 0x34
#define KEY_5 0x35
#define KEY_6 0x36
#define KEY_7 0x37
#define KEY_8 0x38
#define KEY_9 0x39
#define KEY_A 0x41
#define KEY_B 0x42
#define KEY_C 0x43
#define KEY_D 0x44
#define KEY_E 0x45
#define KEY_F 0x46
#define KEY_G 0x47
#define KEY_H 0x48
#define KEY_I 0x49
#define KEY_J 0x4A
#define KEY_K 0x4B
#define KEY_L 0x4C
#define KEY_M 0x4D
#define KEY_N 0x4E
#define KEY_O 0x4F
#define KEY_P 0x50
#define KEY_Q 0x51
#define KEY_R 0x52
#define KEY_S 0x53
#define KEY_T 0x54
#define KEY_U 0x55
#define KEY_V 0x56
#define KEY_W 0x57
#define KEY_X 0x58
#define KEY_Y 0x59
#define KEY_Z 0x5A
#define KEY_SPACE VK_SPACE
#define KEY_LSHIFT VK_SHIFT
#define KEY_ESC VK_ESCAPE
#define KEY_RETURN VK_RETURN
#define KEY_DELETE VK_DELETE
#define KEY_SPACE VK_SPACE
#define KEY_BACKSPACE VK_BACK
#define KEY_ARROW_DOWN VK_DOWN
#define KEY_ARROW_UP VK_UP
#define KEY_ARROW_RIGHT VK_RIGHT
#define KEY_ARROW_LEFT VK_LEFT
#define KEY_CIRCUMFLEX VK_OEM_5

#define MOUSE_BTN_LEFT VK_LBUTTON
#define MOUSE_BTN_RIGHT VK_RBUTTON

#define KEY_DOWN 0x01
#define KEY_RELEASED 0x02
#define KEY_PRESSED 0x03
#define MOUSE_BUTTON_DOWN 0x04
#define MOUSE_BUTTON_PRESSED 0x06
#define MOUSE_BUTTON_RELEASED 0x05
#define MOUSE_WHEEL 0x07
#define TBD_MOUSE_MOVED 0x08
#define MOUSE_DRAGGED 0x09
#define KEY_REPEAT 0x0A

#define CM_FACTORY_END 0x00
#define CM_FACTORY_SOUND 0x01
#define CM_FACTORY_GFX 0x02
#define CM_FACTORY_VIEW 0x03
#define CM_FACTORY_INPUT 0x04
#define CM_FACTORY_GUI 0x05
#define CM_FACTORY_EVENT 0x06
#define CM_FACTORY_LOGIC 0x07
#define CM_FACTROY_CACHE 0x08
#define CM_FACTORY_VRAM 0x09
#define CM_FACTORY_ACTOR 0xA
#define CM_FACTORY_EFFECT 0xB
#define CM_FACTORY_FONT 0xC

#define CM_RENDERPATH_ALBEDO 0x01
#define CM_RENDERPATH_ALBEDO_INSTANCED 0x02
#define CM_RENDERPATH_SHADOWMAP 0x04
#define CM_RENDERPATH_LIGHTING 0x08
#define CM_RENDERPATH_INFO 0x10
#define CM_RENDERPATH_BOUNDING 0x20
#define CM_RENDERPATH_EDITOR 0x40
#define CM_RENDERPATH_SKY 0x80
#define CM_RENDERPATH_ALBEDO_WIRE 0x100
#define CM_RENDERPATH_SHADOWMAP_INSTANCED 0x200
#define CM_RENDERPATH_PICK 0x400
#define CM_RENDERPATH_PARTICLE 0x800
#define CM_RENDERPATH_CNT 12

/*RenderPath_DrawToAlbedo = 1,
    eRenderPath_DrawToShadowMap = 1 << 1,
    eRenderPath_DrawLighting = 1 << 2,
    eRenderPath_DrawEditMode = 1 << 3,
    eRenderPath_DrawPicking = 1 << 4,
    eRenderPath_DrawBounding = 1 << 5,
    eRenderPath_DrawParticles = 1 << 6,
    eRenderPath_DrawDebugInfo = 1 << 7,
    eRenderPath_DrawToAlbedoInstanced = 1 << 8,
    eRenderPath_DrawToShadowMapInstanced = 1 << 9,
    eRenderPath_DrawSky = 1 << 10,
    eRenderPath_NoDraw = 1 << 11,
    eRenderPath_CNT = 12*/


#define ADD_EVENT_LISTENER(_this, function, type) \
    { \
    chimera::EventListener listener = fastdelegate::MakeDelegate(_this, function); \
    chimera::CmGetApp()->VGetEventManager()->VAddEventListener(listener, (chimera::EventType)type); \
    }

#define ADD_EVENT_LISTENER_STATIC(function, type) \
    { \
    chimera::EventListener listener = function; \
    chimera::CmGetApp()->VGetEventManager()->VAddEventListener(listener, (chimera::EventType)type); \
    }

#define REMOVE_EVENT_LISTENER(_this, function, type) \
    { \
    chimera::EventListener listener = fastdelegate::MakeDelegate(_this, function); \
    chimera::CmGetApp()->VGetEventManager()->VRemoveEventListener(listener, (chimera::EventType)type); \
    }

#define REMOVE_EVENT_LISTENER_STATIC(function, type) \
    { \
    chimera::EventListener listener = function; \
    chimera::CmGetApp()->VGetEventManager()->VRemoveEventListener(listener, (chimera::EventType)type); \
    }

#define QUEUE_EVENT(_eventPtr) \
    { \
    chimera::IEventPtr event(_eventPtr); \
    chimera::CmGetApp()->VGetEventManager()->VQueueEvent(event); \
    }

#define TRIGGER_EVENT(_eventPtr) \
    { \
    chimera::IEventPtr event(_eventPtr); \
    chimera::CmGetApp()->VGetEventManager()->VTriggetEvent(event); \
    }

#define QUEUE_EVENT_TSAVE(_eventPtr) \
    { \
    chimera::IEventPtr event(_eventPtr); \
    chimera::CmGetApp()->VGetEventManager()->VQueueEventThreadSave(event); \
    }

#define RETURN_IF_FAILED(__assume__) \
    if(!(__assume__)) { return FALSE; } \

#define HR_RETURN_IF_FAILED(__result__) \
    if(FAILED(__result__)) { return FALSE; } \

#define SAFE_RELEASE(p) { if ( (p) ) { (p)->Release(); (p) = 0; } }
#define SAFE_DELETE(a) if( (a) != NULL ) delete (a); (a) = NULL;
#define SAFE_RESET(a) if( (a) != NULL ) (a.reset()); (a) = NULL;
#define SAFE_ARRAY_DELETE(a) if( (a) != NULL ) delete[] (a); (a) = NULL;

#define TBD_FOR(__iterable) for(auto it = __iterable.begin(); it != __iterable.end(); ++it)
#define TBD_FOR_INT(intVal) for(UINT i = 0; i < intVal; ++i)

#define ACTIVATE_FS 0x01
#define ACTIVATE_VS 0x02
#define ACTIVATE_GS 0x04
#define ACTIVATE_TS 0x08
#define ACTIVATE_ALL (ACTIVATE_FS|ACTIVATE_VS|ACTIVATE_GS|ACTIVATE_TS)

#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define DEGREE_TO_RAD(__deg_) (FLOAT)(((__deg_) / 180.0 * XM_PI))
#define RAD_TO_DEGREE(__deg_) (FLOAT)(((__deg_) * 180.0 / XM_PI))

#define CM_IMAGE_R 0x00
#define CM_IMAGE_RG 0x01
#define CM_IMAGE_RGB 0x02
#define CM_IMAGE_RGBA 0x03
#define CM_IMAGE_sRGBA 0x04

#define CM_STATE_LOADING 0x01
#define CM_STATE_RUNNING 0x02
#define CM_STATE_PAUSED 0x30
