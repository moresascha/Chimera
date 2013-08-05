#pragma once
#include "stdafx.h"
#include "ScreenElement.h"
namespace d3d
{
    class ShaderProgram;
    class EffectChain;
    class RenderTarget;
    class CascadedShadowMapper;
    class PixelShader;
}

namespace tbd
{
    class Query;
    namespace gui
    {
        class GuiTextComponent;
        class D3D_GUI;
    }
    enum SettingType
    {
        eAlbedo = 0,
        eLighting = 1,
    };

    class IGraphicSetting
    {
    protected:
        std::string m_name;
    public:
        IGraphicSetting(LPCSTR settingName) : m_name(settingName) {}
        virtual VOID VRender(VOID) = 0;
        virtual BOOL VOnRestore(UINT w, UINT h) = 0;
        LPCSTR GetName(VOID) { return m_name.c_str(); }
        virtual ~IGraphicSetting(VOID) {}
    };

    class ShaderPathSetting : public IGraphicSetting
    {
    private:
        d3d::ShaderProgram* m_pProgram;
        std::string m_progName;
        UINT m_renderPath;
    public:
        ShaderPathSetting(UINT path, LPCSTR programName, LPCSTR settingName);
        VOID VRender(VOID);
        BOOL VOnRestore(UINT w, UINT h);
    };

    class GloablLightingSetting : public IGraphicSetting
    {
    private:
        d3d::ShaderProgram* m_pGlobalLightProgram;
    public:
        GloablLightingSetting(VOID);
        VOID VRender(VOID);
        BOOL VOnRestore(UINT w, UINT h);
    };

    class LightingSetting : public IGraphicSetting
    {   
    private:
        d3d::ShaderProgram* m_pGlobalLightProgram;
    public:
        LightingSetting(VOID) : IGraphicSetting("Lighting") {}
        VOID VRender(VOID);
        BOOL VOnRestore(UINT w, UINT h) { return TRUE; }
    };

    class CSMSetting : public IGraphicSetting
    {        
    private:
        d3d::CascadedShadowMapper* m_pCSM;
    public:
        CSMSetting(VOID) : IGraphicSetting("CSM"), m_pCSM(NULL) {}
        VOID VRender(VOID);
        BOOL VOnRestore(UINT w, UINT h);
        ~CSMSetting(VOID);
    };

    class PostFXSetting : public IGraphicSetting
    {
    private:
        d3d::EffectChain* m_pEffectChain;
        d3d::RenderTarget* m_pScene;
        d3d::RenderTarget* m_pPreResult;
    public:
        PostFXSetting(VOID) : IGraphicSetting("PostFX"), m_pEffectChain(NULL) {}
        VOID VRender(VOID);
        VOID SetPreResult(d3d::RenderTarget* pPreResult);
        VOID SetScene(d3d::RenderTarget* pScene);
        BOOL VOnRestore(UINT w, UINT h);
        ~PostFXSetting(VOID);
    };

    class BoundingGeoSetting : public ShaderPathSetting
    {
    public:
        BoundingGeoSetting(VOID);
        VOID VRender(VOID);
    };

    class EditModeSetting : public IGraphicSetting
    {
    public:
        EditModeSetting(VOID);
        VOID VRender(VOID);
        BOOL VOnRestore(UINT w, UINT h) { return TRUE; }
    };

    class ProfileSetting : public IGraphicSetting
    {
        friend class GraphicsSettings;
        friend ProfileSetting& operator<<(ProfileSetting& settings, tbd::Query* q);
        
    private:
        IGraphicSetting* m_pSetting;
        std::vector<tbd::Query*> m_pQuerys;
        std::string m_resultAsString;
    public:
        ProfileSetting(IGraphicSetting* setting);
        VOID VRender(VOID);
        BOOL VOnRestore(UINT w, UINT h);
        LPCSTR GetText(VOID);
        ~ProfileSetting(VOID);
    };

    class IGraphicsSettings
    {
    public:
        IGraphicsSettings(VOID);
        virtual VOID VRender(VOID) = 0;
        virtual BOOL VOnRestore(UINT w, UINT h) = 0;
        virtual VOID VOnActivate(VOID) = 0;
        virtual d3d::RenderTarget* VGetResult(VOID) = 0;
        virtual ~IGraphicsSettings(VOID) {}
    };

    class GraphicsSettings : public IGraphicsSettings
    {
    protected:
        std::vector<IGraphicSetting*> m_albedoSettings;
        std::vector<IGraphicSetting*> m_lightSettings;
        IGraphicSetting* m_pPostFX;
        d3d::RenderTarget* m_pScene;
        d3d::RenderTarget* m_pPreResult;
        UINT m_lastW, m_lastH;
    public:
        GraphicsSettings(VOID);
        VOID AddSetting(IGraphicSetting* setting, SettingType type);
        VOID SetPostFX(IGraphicSetting* setting);
        VOID VOnActivate(VOID);
        virtual VOID VRender(VOID);
        virtual BOOL VOnRestore(UINT w, UINT h);

        IGraphicSetting* GetSetting(LPCSTR name);

        virtual d3d::RenderTarget* VGetResult(VOID);
        virtual ~GraphicsSettings(VOID);
    };

    class DefaultGraphicsSettings : public GraphicsSettings
    {
    public:
        DefaultGraphicsSettings(VOID);
    };

    class ProfileGraphicsSettings : public GraphicsSettings
    {
    private:
        tbd::gui::GuiTextComponent* m_pText;
        tbd::gui::D3D_GUI* m_pGui;
    public:
        ProfileGraphicsSettings(VOID);
        VOID VRender(VOID);
        ~ProfileGraphicsSettings(VOID);
    };

    class EditorGraphicsSettings : public DefaultGraphicsSettings
    {
    public:
        EditorGraphicsSettings(VOID);
        VOID VRender(VOID);
    };

    /*
    class DebugGraphicsSettings : public GraphicsSettings
    {
    private:
        d3d::ShaderProgram* m_pDeferredProgram;
        d3d::ShaderProgram* m_pGlobalLight;
        d3d::ShaderProgram* m_pParticlesProgram;
    public:
        DebugGraphicsSettings(VOID);
        BOOL VOnRestore(VOID);
        VOID VRender(VOID);
    };

    class EditorGraphicsSettings : public GraphicsSettings
    {
    private:
        DefaultGraphicsSettings* m_settings;
    public:
        EditorGraphicsSettings(DefaultGraphicsSettings* deco) : m_settings(deco) {}
        VOID VRender(VOID);
        BOOL VOnRestore(VOID);
        d3d::RenderTarget* VGetResult(VOID);
    };
    class WireFrameFilledSettings : public GraphicsSettings
    {
    private:
        DefaultGraphicsSettings* m_settings;
    public:
        WireFrameFilledSettings(DefaultGraphicsSettings* deco) : m_settings(deco) {}
        VOID VRender(VOID);
        BOOL VOnRestore(VOID);
        d3d::RenderTarget* VGetResult(VOID);
    };
    */

    class AlbedoSettings : public GraphicsSettings
    {
    public:
        AlbedoSettings(VOID);
        VOID VRender(VOID);
    };

}
