#pragma once

//#include <D3Dcompiler.h>
#include <D3D11.h>
//#pragma comment(lib, "D3dcompiler.lib")

namespace d3d 
{
    typedef std::string ErrorLog;

#ifdef _DEBUG
    #define CHECK__(__error) \
        if(FAILED(__error)) { \
            LOG_ERROR_A("D3D Error: %d\n", __error); \
        }
#else
    #define CHECK__(__error) __error
#endif

    //static FLOAT m_Color[4] = {1.f, 1.f, 1.f, 1.f};
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

    HRESULT Init(WNDPROC wndProc, CONST HINSTANCE hInstance, CONST LPCWSTR title, UINT width, UINT height);

    ID3D11Device* GetDevice(VOID);

    ID3D11DeviceContext* GetContext(VOID);

    std::string GetShaderError(ID3D10Blob* message);

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
}