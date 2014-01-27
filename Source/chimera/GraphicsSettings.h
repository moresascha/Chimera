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
        virtual void VRender(void);
        virtual bool VOnRestore(uint w, uint h);
        CMShaderProgramDescription* VGetProgramDescription(void) { return &m_desc; }
        virtual ~ShaderPathSetting(void) { }
    };

    class AlbedoSetting : public ShaderPathSetting
    {
    private:
        ShaderPathSetting* m_pInstanced;
    public:
        AlbedoSetting(void);
        void VRender(void);
        bool VOnRestore(uint w, uint h);
        ~AlbedoSetting(void);
    };

	class WireFrameSettings : public ShaderPathSetting
	{
	private:
		std::unique_ptr<IRasterState> m_pWireFrameState;
	public:
		WireFrameSettings(void);
		void VRender(void);
		bool VOnRestore(uint w, uint h);
	};

    class GloablLightingSetting : public ShaderPathSetting
    {
    public:
        GloablLightingSetting(void);
        bool VOnRestore(uint w, uint h);
        void VRender(void);
    };

    //todo settings for all lightsources?
    class LightingSetting : public IGraphicSetting
    {   
    public:
        LightingSetting(void);
        void VRender(void);
        bool VOnRestore(uint w, uint h);
        CMShaderProgramDescription* VGetProgramDescription(void) { return NULL; }
    };

    class CSMSetting : public IGraphicSetting
    {        
    private:
        IEnvironmentLighting* m_pCSM;
    public:
        CSMSetting(void) : IGraphicSetting("CSM"), m_pCSM(NULL) {}
        void VRender(void);
        bool VOnRestore(uint w, uint h);
        CMShaderProgramDescription* VGetProgramDescription(void) { return NULL; }
        ~CSMSetting(void);
    };

    class PostFXSetting : public IPostFXSetting
    {
    private:
        IEffectChain* m_pEffectChain;
        IRenderTarget* m_pSource;
        IRenderTarget* m_pTarget;
    public:
        PostFXSetting(void) : IPostFXSetting("PostFX"),
            m_pEffectChain(NULL), m_pTarget(NULL), m_pSource(NULL) {}
        
        void VRender(void);
        
        void VSetTarget(IRenderTarget* target);
        
        void VSetSource(IRenderTarget* src);

        bool VOnRestore(uint w, uint h);

        IEffectChain* VGetEffectChain(void) { return m_pEffectChain; }

        CMShaderProgramDescription* VGetProgramDescription(void) { return NULL; }

        ~PostFXSetting(void);
    };

    class BoundingGeoSetting : public ShaderPathSetting
    {
    public:
        BoundingGeoSetting(void);
        void VRender(void);
    };

    class EditModeSetting : public ShaderPathSetting
    {
    public:
        EditModeSetting(void);
        bool VOnRestore(uint w, uint h);
    };

    class GuiSetting : public IGraphicSetting
    {
    public:
        GuiSetting(void) : IGraphicSetting("GUI") {}
        void VRender(void);
        bool VOnRestore(uint w, uint h);
        CMShaderProgramDescription* VGetProgramDescription(void) { return NULL; }
        ~GuiSetting(void);
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

        uint m_lastW, m_lastH;
    public:
        GraphicsSettings(void);

        void VAddSetting(std::unique_ptr<IGraphicSetting> settings, GraphicsSettingType type);

        void VSetPostFX(std::unique_ptr<IPostFXSetting> settings);

        IPostFXSetting* VGetPostFX(void) { return m_pPostFX.get(); }

        void VOnActivate(void);

        virtual void VRender(void);

        virtual bool VOnRestore(uint w, uint h);

        virtual IRenderTarget* VGetResult(void);

        virtual ~GraphicsSettings(void);
    };

    class DefaultGraphicsSettings : public GraphicsSettings
    {
    public:
        DefaultGraphicsSettings(void);
    };

    class ProfileGraphicsSettings : public GraphicsSettings
    {
    private:
        /*chimera::gui::GuiTextComponent* m_pText;
        chimera::gui::D3D_GUI* m_pGui; */
    public:
        ProfileGraphicsSettings(void);
        void VRender(void);
        ~ProfileGraphicsSettings(void);
    };

    class EditorGraphicsSettings : public DefaultGraphicsSettings
    {
    public:
        EditorGraphicsSettings(void);
    };

    class BoundingGeoDebugSettings : public GraphicsSettings
    {
    public:
        BoundingGeoDebugSettings(void);
    };

    /*
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
        AlbedoSettings(void);
        void VRender(void);
    };
}
