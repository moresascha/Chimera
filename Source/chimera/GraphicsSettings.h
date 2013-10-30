#pragma once
#include "stdafx.h"
#include "ScreenElement.h"

namespace chimera
{
    class ShaderPathSetting : public IGraphicSetting
    {
    protected:
        IShaderProgram* m_pProgram;
        RenderPath m_renderPath;
        CMShaderProgramDescription m_desc;
        std::string m_programName;
    public:
        ShaderPathSetting(RenderPath path, LPCSTR programName, LPCSTR settingName);
        virtual VOID VRender(VOID);
        virtual BOOL VOnRestore(UINT w, UINT h);
        CMShaderProgramDescription* VGetProgramDescription(VOID) { return &m_desc; }
    };

    class AlbedoSetting : public ShaderPathSetting
    {
    public:
        AlbedoSetting(VOID);
        BOOL VOnRestore(UINT w, UINT h);
    };

    class GloablLightingSetting : public ShaderPathSetting
    {
    public:
        GloablLightingSetting(VOID);
        BOOL VOnRestore(UINT w, UINT h);
        VOID VRender(VOID);
    };

    class LightingSetting : public IGraphicSetting
    {   
    private:
        IShaderProgram* m_pGlobalLightProgram;
    public:
        LightingSetting(VOID) : IGraphicSetting("Lighting") {}
        VOID VRender(VOID);
        BOOL VOnRestore(UINT w, UINT h) { return TRUE; }
    };

    class CSMSetting : public IGraphicSetting
    {        
    private:
        IEnvironmentLighting* m_pCSM;
    public:
        CSMSetting(VOID) : IGraphicSetting("CSM"), m_pCSM(NULL) {}
        VOID VRender(VOID);
        BOOL VOnRestore(UINT w, UINT h);
        CMShaderProgramDescription* VGetProgramDescription(VOID) { return NULL; }
        ~CSMSetting(VOID);
    };

    class PostFXSetting : public IPostFXSetting
    {
    private:
        IEffectChain* m_pEffectChain;
        IRenderTarget* m_pSource;
        IRenderTarget* m_pTarget;
    public:
        PostFXSetting(VOID) : IPostFXSetting("PostFX"),
            m_pEffectChain(NULL), m_pTarget(NULL), m_pSource(NULL) {}
        
        VOID VRender(VOID);
        
        VOID VSetTarget(IRenderTarget* target);
        
        VOID VSetSource(IRenderTarget* src);

        BOOL VOnRestore(UINT w, UINT h);

        CMShaderProgramDescription* VGetProgramDescription(VOID) { return NULL; }

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

    /*
    class ProfileSetting : public IGraphicSetting
    {
        friend class GraphicsSettings;
        friend ProfileSetting& operator<<(ProfileSetting& settings, Query* q);
        
    private:
        IGraphicSetting* m_pSetting;
        //std::vector<chimera::Query*> m_pQuerys;
        std::string m_resultAsString;
    public:
        ProfileSetting(IGraphicSetting* setting);
        VOID VRender(VOID);
        BOOL VOnRestore(UINT w, UINT h);
        LPCSTR GetText(VOID);
        ~ProfileSetting(VOID);
    };*/

    class GraphicsSettings : public IGraphicsSettings
    {
    protected:
        std::vector<std::unique_ptr<IGraphicSetting>> m_albedoSettings;
        std::vector<std::unique_ptr<IGraphicSetting>> m_lightSettings;
        std::unique_ptr<IPostFXSetting> m_pPostFX;
        std::unique_ptr<IRenderTarget> m_pScene;
        //std::unique_ptr<IRenderTarget> m_pPreResult;
        std::unique_ptr<IDepthStencilState> m_pDepthWriteStencilState;
        std::unique_ptr<IDepthStencilState> m_pDepthNoStencilState;
        std::unique_ptr<IRasterState> m_pRasterizerStateFrontFaceSolid;

        UINT m_lastW, m_lastH;
    public:
        GraphicsSettings(VOID);

        VOID VAddSetting(std::unique_ptr<IGraphicSetting> settings, GraphicsSettingType type);

        VOID VSetPostFX(std::unique_ptr<IPostFXSetting> settings);

        VOID VOnActivate(VOID);

        virtual VOID VRender(VOID);

        virtual BOOL VOnRestore(UINT w, UINT h);

        virtual IRenderTarget* VGetResult(VOID);

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
        /*chimera::gui::GuiTextComponent* m_pText;
        chimera::gui::D3D_GUI* m_pGui; */
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

    class BoundingGeoDebugSettings : public GraphicsSettings
    {
    public:
        BoundingGeoDebugSettings(VOID);
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
