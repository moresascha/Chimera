#pragma once
#include "stdafx.h"
#include <D3D11.h>
namespace chimera
{
    namespace d3d
    {

        void __CheckError(HRESULT __error);
#ifdef _DEBUG
#define D3D_SAVE_CALL(__error) __CheckError(__error);
#else
#define D3D_SAVE_CALL(__error) __error
#endif

        extern ID3D11Texture2D *g_pBackBuffer;
        extern ID3D11Texture2D *g_pDepthStencilBuffer;
        extern IDXGISwapChain* g_pSwapChain;
        extern ID3D11Device* g_pDevice;
        extern ID3D11DeviceContext* g_pContext;
        extern ID3D11RenderTargetView* g_pBackBufferView;
        extern ID3D11DepthStencilView* g_pDepthStencilView;
        extern ID3D11DepthStencilState* m_pDepthNoStencilState;
        extern ID3D11DepthStencilState* m_pDepthWriteStencilState;
        extern ID3D11DepthStencilState* m_pDepthCmpStencilState;
        extern ID3D11DepthStencilState* m_pNoDepthNoStencilState;
        extern ID3D11RasterizerState* g_pRasterizerStateBackFaceSolid;
        extern ID3D11RasterizerState* g_pRasterizerStateFrontFaceSolid;
        extern ID3D11RasterizerState* g_pRasterizerStateNoCullingSolid;
        extern ID3D11RasterizerState* g_pRasterizerStateWrireframe;
        extern ID3D11BlendState* g_pBlendStateNoBlending;
        extern ID3D11BlendState* g_pBlendStateBlendAdd;
        extern ID3D11BlendState* g_pBlendStateBlendAlpha;
        extern LPCSTR g_vertexShaderMaxProfile;
        extern LPCSTR g_pixelShaderMaxProfile;
        extern LPCSTR g_geometryShaderMaxProfile;
        extern ID3D11SamplerState* g_pSamplerStates[4];
        extern HWND g_hWnd;

        extern uint g_width;
        extern uint g_height;

        extern uint g_samples;
        extern uint g_quality;

        struct DisplayMode
        {
            DXGI_MODE_DESC mode;

            DisplayMode(void)
            {
                ZeroMemory(&mode, sizeof(DXGI_MODE_DESC));
            }

            DisplayMode(const DisplayMode& cpy)
            {
                mode = cpy.mode;
            }

            DisplayMode(const DXGI_MODE_DESC& cpy)
            {
                mode = cpy;
            }

            DisplayMode& operator=(const DisplayMode& m)
            {
                mode = m.mode;
                return *this;
            }

            DisplayMode& operator=(const DXGI_MODE_DESC& m)
            {
                mode = m;
                return *this;
            }

            void Print(void);
        };

        struct DisplayModeList
        {
            std::vector<DisplayMode> modeList;
        };

        HRESULT Init(WNDPROC wndProc, HINSTANCE hInstance, LPCWSTR title, uint width, uint height);

        ID3D11Device* GetDevice(void);

        ID3D11DeviceContext* GetContext(void);

        std::string GetShaderError(ID3D10Blob* message);

        DXGI_FORMAT GetD3DFormatFromCMFormat(GraphicsFormat Format);

        uint GetWindowHeight(void);

        uint GetWindowWidth(void);

        void BindBackbuffer(void);

        void SetDefaultViewPort(void);

        void SetDefaultStates(void);

        void ClearBackBuffer(const float color[4]);

        void Resize(uint w, uint h);

        void SetFullscreenState(bool fs, uint width = 0, uint height = 0);

        bool GetFullscreenState(void);

        void GetFullscreenSize(uint* width, uint* height);

        void GetDisplayModeList(DisplayModeList& modes);

        LPCSTR GetAdapterName(void);

        DisplayMode GetClosestDisplayMode(const DisplayMode& toMatch);

        void CreateBackbuffer(uint width, uint height);

        void ReleaseBackbuffer(void);

        bool _CreateWindow(WNDPROC wndProc, const HINSTANCE hInstance, const LPCWSTR title, const uint width, const uint height);

        void Release(void);

        class D3DResource
        {
        private:
            uint m_bindFlags;
            D3D11_RESOURCE_MISC_FLAG  m_miscFlags;
            D3D11_USAGE m_usage;
            D3D11_CPU_ACCESS_FLAG m_cpuAccess;

        public:
            D3DResource(void) : m_bindFlags((D3D11_BIND_FLAG)0), m_miscFlags((D3D11_RESOURCE_MISC_FLAG)0), m_usage(D3D11_USAGE_DEFAULT), m_cpuAccess((D3D11_CPU_ACCESS_FLAG)0) {}

            virtual uint GetBindflags(void)
            {
                return m_bindFlags;
            }

            virtual D3D11_USAGE GetUsage(void)
            {
                return m_usage;
            }

            virtual D3D11_CPU_ACCESS_FLAG GetCPUAccess(void)
            {
                return m_cpuAccess;
            }

            virtual void SetBindflags(uint flags)
            {
                m_bindFlags = flags;
            }

            virtual void SetUsage(D3D11_USAGE usage)
            {
                m_usage = usage;
            }

            virtual void SetCPUAccess(D3D11_CPU_ACCESS_FLAG cpua)
            {
                m_cpuAccess = cpua;
            }

            void SetMiscflags(D3D11_RESOURCE_MISC_FLAG flags)
            {
                m_miscFlags = flags;
            }

            D3D11_RESOURCE_MISC_FLAG GetMiscflags(void)
            {
                return m_miscFlags;
            }

            virtual ~D3DResource(void) {}
        };
    }
}
