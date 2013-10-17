#pragma once
#include "stdafx.h"
#include <D3D11.h>
namespace chimera
{
    namespace d3d
    {

        VOID __CheckError(HRESULT __error);
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

        extern UINT g_width;
        extern UINT g_height;

        extern UINT g_samples;
        extern UINT g_quality;

        struct DisplayMode
        {
            DXGI_MODE_DESC mode;

            DisplayMode(VOID)
            {
                ZeroMemory(&mode, sizeof(DXGI_MODE_DESC));
            }

            DisplayMode(CONST DisplayMode& cpy)
            {
                mode = cpy.mode;
            }

            DisplayMode(CONST DXGI_MODE_DESC& cpy)
            {
                mode = cpy;
            }

            DisplayMode& operator=(CONST DisplayMode& m)
            {
                mode = m.mode;
                return *this;
            }

            DisplayMode& operator=(CONST DXGI_MODE_DESC& m)
            {
                mode = m;
                return *this;
            }

            VOID Print(VOID);
        };

        struct DisplayModeList
        {
            std::vector<DisplayMode> modeList;
        };

        HRESULT Init(WNDPROC wndProc, HINSTANCE hInstance, LPCWSTR title, UINT width, UINT height);

        ID3D11Device* GetDevice(VOID);

        ID3D11DeviceContext* GetContext(VOID);

        std::string GetShaderError(ID3D10Blob* message);

        DXGI_FORMAT GetD3DFormatFromCMFormat(GraphicsFormat Format);

        UINT GetWindowHeight(VOID);

        UINT GetWindowWidth(VOID);

        VOID BindBackbuffer(VOID);

        VOID SetDefaultViewPort(VOID);

        VOID SetDefaultStates(VOID);

        VOID ClearBackBuffer(CONST FLOAT color[4]);

        VOID Resize(UINT w, UINT h);

        VOID SetFullscreenState(BOOL fs, UINT width = 0, UINT height = 0);

        BOOL GetFullscreenState(VOID);

        VOID GetFullscreenSize(UINT* width, UINT* height);

        VOID GetDisplayModeList(DisplayModeList& modes);

        LPCSTR GetAdapterName(VOID);

        DisplayMode GetClosestDisplayMode(CONST DisplayMode& toMatch);

        VOID CreateBackbuffer(UINT width, UINT height);

        VOID ReleaseBackbuffer(VOID);

        BOOL _CreateWindow(WNDPROC wndProc, CONST HINSTANCE hInstance, CONST LPCWSTR title, CONST UINT width, CONST UINT height);

        VOID Release(VOID);

        class D3DResource
        {
        private:
            UINT m_bindFlags;
            D3D11_RESOURCE_MISC_FLAG  m_miscFlags;
            D3D11_USAGE m_usage;
            D3D11_CPU_ACCESS_FLAG m_cpuAccess;

        public:
            D3DResource(VOID) : m_bindFlags((D3D11_BIND_FLAG)0), m_miscFlags((D3D11_RESOURCE_MISC_FLAG)0), m_usage(D3D11_USAGE_DEFAULT), m_cpuAccess((D3D11_CPU_ACCESS_FLAG)0) {}

            virtual UINT GetBindflags(VOID)
            {
                return m_bindFlags;
            }

            virtual D3D11_USAGE GetUsage(VOID)
            {
                return m_usage;
            }

            virtual D3D11_CPU_ACCESS_FLAG GetCPUAccess(VOID)
            {
                return m_cpuAccess;
            }

            virtual VOID SetBindflags(UINT flags)
            {
                m_bindFlags = flags;
            }

            virtual VOID SetUsage(D3D11_USAGE usage)
            {
                m_usage = usage;
            }

            virtual VOID SetCPUAccess(D3D11_CPU_ACCESS_FLAG cpua)
            {
                m_cpuAccess = cpua;
            }

            VOID SetMiscflags(D3D11_RESOURCE_MISC_FLAG flags)
            {
                m_miscFlags = flags;
            }

            D3D11_RESOURCE_MISC_FLAG GetMiscflags(VOID)
            {
                return m_miscFlags;
            }

            virtual ~D3DResource(VOID) {}
        };
    }
}
