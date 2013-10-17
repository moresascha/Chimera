#include "D3DGraphics.h"

#include <Initguid.h>
#include <dxgidebug.h>

#ifdef _DEBUG
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#else
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#endif

namespace chimera
{
    namespace d3d
    {

        LPCSTR GetErrorAsString(HRESULT error)
        {
            switch(error)
            {
            case D3D11_ERROR_FILE_NOT_FOUND : return "D3D11_ERROR_FILE_NOT_FOUND";
            case D3D11_ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS : return "D3D11_ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS";
            case D3D11_ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS : return "D3D11_ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS";
            case D3D11_ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD : return "D3D11_ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD";
            case E_FAIL : return "E_FAIL";
            case E_INVALIDARG : return "E_INVALIDARG";
            case E_OUTOFMEMORY : return "E_OUTOFMEMORY";
            case S_FALSE : return "S_FALSE";
            case S_OK : return "S_OK";
            }
            return "";
        }

        VOID __CheckError(HRESULT __error)
        {
            if(FAILED(__error)) 
            {
                LOG_CRITICAL_ERROR_A("%s\n", GetErrorAsString(__error));
            }
        }

        IDXGIFactory* g_pDXGIFactory = 0;
        IDXGIAdapter* g_pAdapter = 0;

        ID3D11Texture2D *g_pBackBuffer = 0;
        ID3D11Texture2D *g_pDepthStencilBuffer = 0;
        IDXGISwapChain* g_pSwapChain = 0;
        ID3D11Device* g_pDevice = 0;
        ID3D11DeviceContext* g_pContext = 0;
        ID3D11RenderTargetView* g_pBackBufferView = 0;
        ID3D11DepthStencilView* g_pDepthStencilView = 0;
        ID3D11DepthStencilState* m_pDepthNoStencilState = 0;
        ID3D11DepthStencilState* m_pDepthCmpStencilState = 0;
        ID3D11DepthStencilState* m_pDepthWriteStencilState = 0;
        ID3D11DepthStencilState* m_pNoDepthNoStencilState = 0;
        ID3D11RasterizerState* g_pRasterizerStateFrontFaceSolid = 0;
        ID3D11RasterizerState* g_pRasterizerStateNoCullingSolid = 0;
        ID3D11RasterizerState* g_pRasterizerStateBackFaceSolid = 0;
        ID3D11RasterizerState* g_pRasterizerStateWrireframe = 0;
        ID3D11BlendState* g_pBlendStateNoBlending = 0;
        ID3D11BlendState* g_pBlendStateBlendAdd = 0;
        ID3D11BlendState* g_pBlendStateBlendAlpha = 0;
        ID3D11SamplerState* g_pSamplerStates[4];
        LPCSTR g_vertexShaderMaxProfile;
        LPCSTR g_pixelShaderMaxProfile;
        LPCSTR g_geometryShaderMaxProfile;
        HWND g_hWnd;
        UINT g_width;
        UINT g_height;
        UINT g_samples = 1;
        UINT g_quality = 0;

        UINT flags = 0;

        std::string* g_adapterName = 0;

        LPCSTR DXGI_FORMAT_STR[] = 
        {    
            "DXGI_FORMAT_UNKNOWN",
            "DXGI_FORMAT_R32G32B32A32_TYPELESS",
            "DXGI_FORMAT_R32G32B32A32_FLOAT",
            "DXGI_FORMAT_R32G32B32A32_UINT",
            "DXGI_FORMAT_R32G32B32A32_SINT",
            "DXGI_FORMAT_R32G32B32_TYPELESS",
            "DXGI_FORMAT_R32G32B32_FLOAT",
            "DXGI_FORMAT_R32G32B32_UINT",
            "DXGI_FORMAT_R32G32B32_SINT",
            "DXGI_FORMAT_R16G16B16A16_TYPELESS",
            "DXGI_FORMAT_R16G16B16A16_FLOAT",
            "DXGI_FORMAT_R16G16B16A16_UNORM",
            "DXGI_FORMAT_R16G16B16A16_UINT",
            "DXGI_FORMAT_R16G16B16A16_SNORM",
            "DXGI_FORMAT_R16G16B16A16_SINT",
            "DXGI_FORMAT_R32G32_TYPELESS",
            "DXGI_FORMAT_R32G32_FLOAT",
            "DXGI_FORMAT_R32G32_UINT",
            "DXGI_FORMAT_R32G32_SINT",
            "DXGI_FORMAT_R32G8X24_TYPELESS",
            "DXGI_FORMAT_D32_FLOAT_S8X24_UINT",
            "DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS",
            "DXGI_FORMAT_X32_TYPELESS_G8X24_UINT",
            "DXGI_FORMAT_R10G10B10A2_TYPELESS",
            "DXGI_FORMAT_R10G10B10A2_UNORM",
            "DXGI_FORMAT_R10G10B10A2_UINT",
            "DXGI_FORMAT_R11G11B10_FLOAT",
            "DXGI_FORMAT_R8G8B8A8_TYPELESS",
            "DXGI_FORMAT_R8G8B8A8_UNORM",
            "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB",
            "DXGI_FORMAT_R8G8B8A8_UINT",
            "DXGI_FORMAT_R8G8B8A8_SNORM",
            "DXGI_FORMAT_R8G8B8A8_SINT",
            "DXGI_FORMAT_R16G16_TYPELESS",
            "DXGI_FORMAT_R16G16_FLOAT",
            "DXGI_FORMAT_R16G16_UNORM",
            "DXGI_FORMAT_R16G16_UINT",
            "DXGI_FORMAT_R16G16_SNORM",
            "DXGI_FORMAT_R16G16_SINT",
            "DXGI_FORMAT_R32_TYPELESS",
            "DXGI_FORMAT_D32_FLOAT",
            "DXGI_FORMAT_R32_FLOAT",
            "DXGI_FORMAT_R32_UINT",
            "DXGI_FORMAT_R32_SINT",
            "DXGI_FORMAT_R24G8_TYPELESS",
            "DXGI_FORMAT_D24_UNORM_S8_UINT",
            "DXGI_FORMAT_R24_UNORM_X8_TYPELESS",
            "DXGI_FORMAT_X24_TYPELESS_G8_UINT",
            "DXGI_FORMAT_R8G8_TYPELESS",
            "DXGI_FORMAT_R8G8_UNORM",
            "DXGI_FORMAT_R8G8_UINT",
            "DXGI_FORMAT_R8G8_SNORM",
            "DXGI_FORMAT_R8G8_SINT",
            "DXGI_FORMAT_R16_TYPELESS",
            "DXGI_FORMAT_R16_FLOAT",
            "DXGI_FORMAT_D16_UNORM",
            "DXGI_FORMAT_R16_UNORM",
            "DXGI_FORMAT_R16_UINT",
            "DXGI_FORMAT_R16_SNORM",
            "DXGI_FORMAT_R16_SINT",
            "DXGI_FORMAT_R8_TYPELESS",
            "DXGI_FORMAT_R8_UNORM",
            "DXGI_FORMAT_R8_UINT",
            "DXGI_FORMAT_R8_SNORM",
            "DXGI_FORMAT_R8_SINT",
            "DXGI_FORMAT_A8_UNORM",
            "DXGI_FORMAT_R1_UNORM",
            "DXGI_FORMAT_R9G9B9E5_SHAREDEXP",
            "DXGI_FORMAT_R8G8_B8G8_UNORM",
            "DXGI_FORMAT_G8R8_G8B8_UNORM",
            "DXGI_FORMAT_BC1_TYPELESS",
            "DXGI_FORMAT_BC1_UNORM",
            "DXGI_FORMAT_BC1_UNORM_SRGB",
            "DXGI_FORMAT_BC2_TYPELESS",
            "DXGI_FORMAT_BC2_UNORM",
            "DXGI_FORMAT_BC2_UNORM_SRGB",
            "DXGI_FORMAT_BC3_TYPELESS",
            "DXGI_FORMAT_BC3_UNORM",
            "DXGI_FORMAT_BC3_UNORM_SRGB",
            "DXGI_FORMAT_BC4_TYPELESS",
            "DXGI_FORMAT_BC4_UNORM",
            "DXGI_FORMAT_BC4_SNORM",
            "DXGI_FORMAT_BC5_TYPELESS",
            "DXGI_FORMAT_BC5_UNORM",
            "DXGI_FORMAT_BC5_SNORM",
            "DXGI_FORMAT_B5G6R5_UNORM",
            "DXGI_FORMAT_B5G5R5A1_UNORM",
            "DXGI_FORMAT_B8G8R8A8_UNORM",
            "DXGI_FORMAT_B8G8R8X8_UNORM",
            "DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM",
            "DXGI_FORMAT_B8G8R8A8_TYPELESS",
            "DXGI_FORMAT_B8G8R8A8_UNORM_SRGB",
            "DXGI_FORMAT_B8G8R8X8_TYPELESS",
            "DXGI_FORMAT_B8G8R8X8_UNORM_SRGB",
            "DXGI_FORMAT_BC6H_TYPELESS",
            "DXGI_FORMAT_BC6H_UF16",
            "DXGI_FORMAT_BC6H_SF16",
            "DXGI_FORMAT_BC7_TYPELESS",
            "DXGI_FORMAT_BC7_UNORM",
            "DXGI_FORMAT_BC7_UNORM_SRGB",
            "DXGI_FORMAT_AYUV",
            "DXGI_FORMAT_Y410",
            "DXGI_FORMAT_Y416",
            "DXGI_FORMAT_NV12",
            "DXGI_FORMAT_P010",
            "DXGI_FORMAT_P016",
            "DXGI_FORMAT_420_OPAQUE",
            "DXGI_FORMAT_YUY2",
            "DXGI_FORMAT_Y210",
            "DXGI_FORMAT_Y216",
            "DXGI_FORMAT_NV11",
            "DXGI_FORMAT_AI44",
            "DXGI_FORMAT_IA44",
            "DXGI_FORMAT_P8",
            "DXGI_FORMAT_A8P8",
            "DXGI_FORMAT_B4G4R4A4_UNORM",
            "DXGI_FORMAT_FORCE_UINT" 
        };

        DXGI_FORMAT d3dFormatMap[] = 
        {
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_R32G32B32A32_TYPELESS,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_UINT,
            DXGI_FORMAT_R32G32B32A32_SINT,
            DXGI_FORMAT_R32G32B32_TYPELESS,
            DXGI_FORMAT_R32G32B32_FLOAT,
            DXGI_FORMAT_R32G32B32_UINT,
            DXGI_FORMAT_R32G32B32_SINT,
            DXGI_FORMAT_R16G16B16A16_TYPELESS,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R16G16B16A16_UNORM,
            DXGI_FORMAT_R16G16B16A16_UINT,
            DXGI_FORMAT_R16G16B16A16_SNORM,
            DXGI_FORMAT_R16G16B16A16_SINT,
            DXGI_FORMAT_R32G32_TYPELESS,
            DXGI_FORMAT_R32G32_FLOAT,
            DXGI_FORMAT_R32G32_UINT,
            DXGI_FORMAT_R32G32_SINT,
            DXGI_FORMAT_R32G8X24_TYPELESS,
            DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
            DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
            DXGI_FORMAT_R10G10B10A2_TYPELESS,
            DXGI_FORMAT_R10G10B10A2_UNORM,
            DXGI_FORMAT_R10G10B10A2_UINT,
            DXGI_FORMAT_R11G11B10_FLOAT,
            DXGI_FORMAT_R8G8B8A8_TYPELESS,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
            DXGI_FORMAT_R8G8B8A8_UINT,
            DXGI_FORMAT_R8G8B8A8_SNORM,
            DXGI_FORMAT_R8G8B8A8_SINT,
            DXGI_FORMAT_R16G16_TYPELESS,
            DXGI_FORMAT_R16G16_FLOAT,
            DXGI_FORMAT_R16G16_UNORM,
            DXGI_FORMAT_R16G16_UINT,
            DXGI_FORMAT_R16G16_SNORM,
            DXGI_FORMAT_R16G16_SINT,
            DXGI_FORMAT_R32_TYPELESS,
            DXGI_FORMAT_D32_FLOAT,
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32_UINT,
            DXGI_FORMAT_R32_SINT,
            DXGI_FORMAT_R24G8_TYPELESS,
            DXGI_FORMAT_D24_UNORM_S8_UINT,
            DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
            DXGI_FORMAT_X24_TYPELESS_G8_UINT,
            DXGI_FORMAT_R8G8_TYPELESS,
            DXGI_FORMAT_R8G8_UNORM,
            DXGI_FORMAT_R8G8_UINT,
            DXGI_FORMAT_R8G8_SNORM,
            DXGI_FORMAT_R8G8_SINT,
            DXGI_FORMAT_R16_TYPELESS,
            DXGI_FORMAT_R16_FLOAT,
            DXGI_FORMAT_D16_UNORM,
            DXGI_FORMAT_R16_UNORM,
            DXGI_FORMAT_R16_UINT,
            DXGI_FORMAT_R16_SNORM,
            DXGI_FORMAT_R16_SINT,
            DXGI_FORMAT_R8_TYPELESS,
            DXGI_FORMAT_R8_UNORM,
            DXGI_FORMAT_R8_UINT,
            DXGI_FORMAT_R8_SNORM,
            DXGI_FORMAT_R8_SINT,
            DXGI_FORMAT_A8_UNORM,
            DXGI_FORMAT_R1_UNORM,
            DXGI_FORMAT_R9G9B9E5_SHAREDEXP,
            DXGI_FORMAT_R8G8_B8G8_UNORM,
            DXGI_FORMAT_G8R8_G8B8_UNORM,
            DXGI_FORMAT_BC1_TYPELESS,
            DXGI_FORMAT_BC1_UNORM,
            DXGI_FORMAT_BC1_UNORM_SRGB,
            DXGI_FORMAT_BC2_TYPELESS,
            DXGI_FORMAT_BC2_UNORM,
            DXGI_FORMAT_BC2_UNORM_SRGB,
            DXGI_FORMAT_BC3_TYPELESS,
            DXGI_FORMAT_BC3_UNORM,
            DXGI_FORMAT_BC3_UNORM_SRGB,
            DXGI_FORMAT_BC4_TYPELESS,
            DXGI_FORMAT_BC4_UNORM,
            DXGI_FORMAT_BC4_SNORM,
            DXGI_FORMAT_BC5_TYPELESS,
            DXGI_FORMAT_BC5_UNORM,
            DXGI_FORMAT_BC5_SNORM,
            DXGI_FORMAT_B5G6R5_UNORM,
            DXGI_FORMAT_B5G5R5A1_UNORM,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            DXGI_FORMAT_B8G8R8X8_UNORM,
            DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM,
            DXGI_FORMAT_B8G8R8A8_TYPELESS,
            DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
            DXGI_FORMAT_B8G8R8X8_TYPELESS,
            DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,
            DXGI_FORMAT_BC6H_TYPELESS,
            DXGI_FORMAT_BC6H_UF16,
            DXGI_FORMAT_BC6H_SF16,
            DXGI_FORMAT_BC7_TYPELESS,
            DXGI_FORMAT_BC7_UNORM,
            DXGI_FORMAT_BC7_UNORM_SRGB,
            DXGI_FORMAT_AYUV,
            DXGI_FORMAT_Y410,
            DXGI_FORMAT_Y416,
            DXGI_FORMAT_NV12,
            DXGI_FORMAT_P010,
            DXGI_FORMAT_P016,
            DXGI_FORMAT_420_OPAQUE,
            DXGI_FORMAT_YUY2,
            DXGI_FORMAT_Y210,
            DXGI_FORMAT_Y216,
            DXGI_FORMAT_NV11,
            DXGI_FORMAT_AI44,
            DXGI_FORMAT_IA44,
            DXGI_FORMAT_P8,
            DXGI_FORMAT_A8P8,
            DXGI_FORMAT_B4G4R4A4_UNORM,
            DXGI_FORMAT_FORCE_UINT
        };

        VOID DisplayMode::Print(VOID)
        {
            DEBUG_OUT_A("Widht=%d Height=%d Refreshrate=(%d %d) Format=%s\n", 
                mode.Width, mode.Height, mode.RefreshRate.Numerator, mode.RefreshRate.Denominator, DXGI_FORMAT_STR[mode.Format]);
        }

        HRESULT Init(WNDPROC wndProc, CONST HINSTANCE hInstance, LPCWSTR title, UINT width, UINT height) 
        {
            DXGI_MODE_DESC md;
            ZeroMemory(&md, sizeof(DXGI_MODE_DESC));
            md.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            md.Height = height;
            md.Width = width;
            md.RefreshRate.Denominator = 60000;
            md.RefreshRate.Numerator = 1000;

            D3D_SAVE_CALL(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&g_pDXGIFactory));
            D3D_SAVE_CALL(g_pDXGIFactory->EnumAdapters(0, &g_pAdapter));

            DXGI_ADAPTER_DESC adapterDesc;
            g_pAdapter->GetDesc(&adapterDesc);

            WCHAR* d = adapterDesc.Description;
            std::wstring ws = d;
            g_adapterName = new std::string(ws.begin(), ws.end());

            DisplayMode fit = GetClosestDisplayMode(md);

            width = fit.mode.Width;
            height = fit.mode.Height;

            _CreateWindow(wndProc, hInstance, title, width, height);

            g_width = width;
            g_height = height;

            DXGI_SWAP_CHAIN_DESC desc;
            ZeroMemory(&desc, sizeof(DXGI_SWAP_CHAIN_DESC));
            desc.BufferCount = 1;
            desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            desc.BufferDesc.RefreshRate.Numerator = fit.mode.RefreshRate.Numerator;
            desc.BufferDesc.RefreshRate.Denominator = fit.mode.RefreshRate.Denominator;
            desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; 
            desc.OutputWindow = g_hWnd;
            desc.SampleDesc.Count = g_samples;
            desc.SampleDesc.Quality = g_quality;
            desc.Windowed = TRUE;
            desc.BufferDesc.Height = height;
            desc.BufferDesc.Width = width;
            desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

            D3D_FEATURE_LEVEL level;

            D3D_FEATURE_LEVEL pLevel[] = 
            {
                D3D_FEATURE_LEVEL_11_0,
                D3D_FEATURE_LEVEL_10_1,
                D3D_FEATURE_LEVEL_10_0,
            };

#ifdef _DEBUG
           //flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

            D3D_SAVE_CALL(D3D11CreateDeviceAndSwapChain(NULL,
                D3D_DRIVER_TYPE_HARDWARE,
                NULL,
                flags,
                pLevel,
                3,
                D3D11_SDK_VERSION,
                &desc,
                &g_pSwapChain,
                &g_pDevice,
                &level,
                &g_pContext));

            switch(level) 
            {
            case D3D_FEATURE_LEVEL_10_0 : 
                {
                    g_vertexShaderMaxProfile = "vs_4_0";
                    g_pixelShaderMaxProfile = "ps_4_0";
                    g_geometryShaderMaxProfile = "gs_4_0.0";
                    DEBUG_OUT("Using Direct3D 10.0\n"); 
                    break;
                }
            case D3D_FEATURE_LEVEL_10_1 : 
                {
                    g_vertexShaderMaxProfile = "vs_4_0";
                    g_pixelShaderMaxProfile = "ps_4_0";
                    g_geometryShaderMaxProfile = "gs_4_0";
                    DEBUG_OUT("Using Direct3D 10.1\n");
                    break;
                }
            case D3D_FEATURE_LEVEL_11_0 : 
                {
                    g_vertexShaderMaxProfile = "vs_5_0";
                    g_geometryShaderMaxProfile = "gs_5_0";
                    g_pixelShaderMaxProfile = "ps_5_0";
                    DEBUG_OUT("Using Direct3D 11.0\n"); 
                    break;
                }
            } 

            CreateBackbuffer(width, height);
                        
            D3D11_DEPTH_STENCIL_DESC dsDesc;
            ZeroMemory(&dsDesc, sizeof(D3D11_DEPTH_STENCIL_DESC));
            dsDesc.DepthEnable = TRUE;
            dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
            dsDesc.DepthFunc = D3D11_COMPARISON_LESS_EQUAL;
            dsDesc.StencilEnable = FALSE;

            //D3D_SAVE_CALL(g_pDevice->CreateDepthStencilState(&dsDesc, &m_pDepthNoStencilState));

            dsDesc.StencilEnable = TRUE;
            dsDesc.StencilWriteMask = 0xFF;
            dsDesc.StencilReadMask = 0xFF;

            dsDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
            dsDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
            dsDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_INCR;
            dsDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;

            dsDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
            dsDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
            dsDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_INCR; 
            dsDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;

           // D3D_SAVE_CALL(g_pDevice->CreateDepthStencilState(&dsDesc, &m_pDepthWriteStencilState));

            dsDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
            dsDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
            dsDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_INCR;
            dsDesc.FrontFace.StencilFunc = D3D11_COMPARISON_EQUAL;

            dsDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
            dsDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
            dsDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_INCR;
            dsDesc.BackFace.StencilFunc = D3D11_COMPARISON_EQUAL;
            //D3D_SAVE_CALL(g_pDevice->CreateDepthStencilState(&dsDesc, &m_pDepthCmpStencilState));

            dsDesc.DepthEnable = FALSE;
            dsDesc.StencilEnable = FALSE;
           // D3D_SAVE_CALL(g_pDevice->CreateDepthStencilState(&dsDesc, &m_pNoDepthNoStencilState));

            D3D11_BLEND_DESC blendDesc;
            ZeroMemory(&blendDesc, sizeof(D3D11_BLEND_DESC));
            for(int i = 0; i < 8; ++i)
            {
                blendDesc.RenderTarget[i].BlendEnable = FALSE;
                blendDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
            }
           // D3D_SAVE_CALL(g_pDevice->CreateBlendState(&blendDesc, &g_pBlendStateNoBlending));

            ZeroMemory(&blendDesc, sizeof(D3D11_BLEND_DESC));
            for(int i = 0; i < 8; ++i)
            {
                blendDesc.RenderTarget[i].BlendEnable = TRUE;
                blendDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

                blendDesc.RenderTarget[i].BlendOp = D3D11_BLEND_OP_ADD;
                blendDesc.RenderTarget[i].BlendOpAlpha = D3D11_BLEND_OP_ADD;

                blendDesc.RenderTarget[i].DestBlend = D3D11_BLEND_ONE;
                blendDesc.RenderTarget[i].DestBlendAlpha = D3D11_BLEND_ONE;

                blendDesc.RenderTarget[i].SrcBlend = D3D11_BLEND_SRC_ALPHA;
                blendDesc.RenderTarget[i].SrcBlendAlpha = D3D11_BLEND_ONE;  
            }

            //D3D_SAVE_CALL(g_pDevice->CreateBlendState(&blendDesc, &g_pBlendStateBlendAdd));

            for(int i = 0; i < 8; ++i)
            {
                blendDesc.RenderTarget[i].BlendEnable = TRUE;
                blendDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

                blendDesc.RenderTarget[i].BlendOp = D3D11_BLEND_OP_ADD;
                blendDesc.RenderTarget[i].BlendOpAlpha = D3D11_BLEND_OP_ADD;

                blendDesc.RenderTarget[i].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
                blendDesc.RenderTarget[i].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;

                blendDesc.RenderTarget[i].SrcBlend = D3D11_BLEND_SRC_ALPHA;
                blendDesc.RenderTarget[i].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;  
            }

            //D3D_SAVE_CALL(g_pDevice->CreateBlendState(&blendDesc, &g_pBlendStateBlendAlpha));

            D3D11_RASTERIZER_DESC rasterDesc;
            ZeroMemory(&rasterDesc, sizeof(D3D11_RASTERIZER_DESC));
            rasterDesc.CullMode = D3D11_CULL_BACK;
            rasterDesc.FillMode = D3D11_FILL_SOLID;
            rasterDesc.DepthClipEnable = TRUE;
            rasterDesc.FrontCounterClockwise = TRUE;
            rasterDesc.MultisampleEnable = FALSE;
            rasterDesc.AntialiasedLineEnable = FALSE;
            //D3D_SAVE_CALL(g_pDevice->CreateRasterizerState(&rasterDesc, &g_pRasterizerStateFrontFaceSolid));

            rasterDesc.FillMode = D3D11_FILL_WIREFRAME;
            rasterDesc.CullMode = D3D11_CULL_NONE;
            //D3D_SAVE_CALL(g_pDevice->CreateRasterizerState(&rasterDesc, &g_pRasterizerStateWrireframe));

            rasterDesc.FillMode = D3D11_FILL_SOLID;
            rasterDesc.CullMode = D3D11_CULL_NONE;
            //D3D_SAVE_CALL(g_pDevice->CreateRasterizerState(&rasterDesc, &g_pRasterizerStateNoCullingSolid));

            rasterDesc.CullMode = D3D11_CULL_FRONT;
            //D3D_SAVE_CALL(g_pDevice->CreateRasterizerState(&rasterDesc, &g_pRasterizerStateBackFaceSolid));

            D3D11_SAMPLER_DESC sDesc;
            ZeroMemory(&sDesc, sizeof(D3D11_SAMPLER_DESC));

            sDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            sDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
            sDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
            sDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
            sDesc.MipLODBias = 0.0f;
            sDesc.MaxAnisotropy = 1;
            sDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
            sDesc.BorderColor[0] = 0;
            sDesc.BorderColor[1] = 0;
            sDesc.BorderColor[2] = 0;
            sDesc.BorderColor[3] = 0;
            sDesc.MinLOD = 0;
            sDesc.MaxLOD = D3D11_FLOAT32_MAX;

            D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateSamplerState(&sDesc, &g_pSamplerStates[0]));

            sDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            sDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            sDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            sDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            sDesc.MaxLOD = D3D11_FLOAT32_MAX;

            D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateSamplerState(&sDesc, &g_pSamplerStates[1]));

            sDesc.Filter = D3D11_FILTER_ANISOTROPIC;
            sDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
            sDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
            sDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
            sDesc.MaxAnisotropy = 16;
            sDesc.ComparisonFunc = D3D11_COMPARISON_ALWAYS;
            sDesc.MaxLOD = D3D11_FLOAT32_MAX;

            D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateSamplerState(&sDesc, &g_pSamplerStates[2]));

            sDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
            sDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            sDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            sDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            sDesc.MaxLOD = D3D11_FLOAT32_MAX;

            D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateSamplerState(&sDesc, &g_pSamplerStates[3]));
           
            SetDefaultStates();

            return TRUE;
        }

        LPCSTR GetAdapterName(VOID)
        {
            return g_adapterName->c_str();
        }

        VOID GetDisplayModeList(DisplayModeList& modes)
        {
            IDXGIOutput* output;
            D3D_SAVE_CALL(g_pAdapter->EnumOutputs(0, &output));

            UINT n;
            output->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &n, NULL);

            DXGI_MODE_DESC* mds = new DXGI_MODE_DESC[n];
            output->GetDisplayModeList(DXGI_FORMAT_R8G8B8A8_UNORM, 0, &n, mds);

            TBD_FOR_INT(n)
            {
                DisplayMode m = mds[i];
                modes.modeList.push_back(m);
            }
            SAFE_RELEASE(output);
            SAFE_ARRAY_DELETE(mds);
        }

        DisplayMode GetClosestDisplayMode(CONST DisplayMode& mode)
        {
            IDXGIOutput* output;
            D3D_SAVE_CALL(g_pAdapter->EnumOutputs(0, &output));
            DisplayMode dm;
            D3D_SAVE_CALL(output->FindClosestMatchingMode(&mode.mode, &dm.mode, g_pDevice));
            SAFE_RELEASE(output);
            return dm;
        }

        UINT GetWindowHeight(VOID)
        {
            return g_height;
        }

        UINT GetWindowWidth(VOID)
        {
            return g_width;
        }

        VOID GetFullscreenSize(UINT* width, UINT* height)
        {
            RECT r;
            GetWindowRect(GetDesktopWindow(), &r);
            *height = r.bottom;
            *width = r.right;
        }

        VOID ReleaseBackbuffer(VOID) 
        {
            g_pContext->OMSetRenderTargets(0, 0, 0);
            SAFE_RELEASE(g_pDepthStencilBuffer);
            SAFE_RELEASE(g_pDepthStencilView);
            SAFE_RELEASE(g_pBackBuffer);
            SAFE_RELEASE(g_pBackBufferView);
        }

        VOID BindBackbuffer(VOID) 
        {
            g_pContext->OMSetRenderTargets(1, &g_pBackBufferView, g_pDepthStencilView);
            //chimera::SetDefaultViewPort();
        }

        VOID ClearBackBuffer(CONST FLOAT color[4])
        {
            g_pContext->ClearRenderTargetView(g_pBackBufferView, color);
            g_pContext->ClearDepthStencilView(g_pDepthStencilView, D3D11_CLEAR_STENCIL | D3D11_CLEAR_DEPTH, 1, 1);
        }

        VOID SetDefaultViewPort(VOID)
        {
            D3D11_VIEWPORT port;
            port.MinDepth = 0;
            port.MaxDepth = 1;
            port.TopLeftX = 0;
            port.TopLeftY = 0;
            port.Width = (FLOAT)g_width;
            port.Height = (FLOAT)g_height;

            g_pContext->RSSetViewports(1, &port);
        }

        VOID CreateBackbuffer(UINT width, UINT height) 
        {
            g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&g_pBackBuffer);

            D3D_SAVE_CALL(g_pDevice->CreateRenderTargetView(g_pBackBuffer, NULL, &g_pBackBufferView));

            D3D11_TEXTURE2D_DESC dsbDesc;
            ZeroMemory(&dsbDesc, sizeof(D3D11_TEXTURE2D_DESC));
            dsbDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
            dsbDesc.MipLevels = 1;
            dsbDesc.Width = width;
            dsbDesc.Height = height;
            dsbDesc.SampleDesc.Quality = g_quality;
            dsbDesc.SampleDesc.Count = g_samples;
            dsbDesc.Usage = D3D11_USAGE_DEFAULT;
            dsbDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
            dsbDesc.ArraySize = 1;
            dsbDesc.MiscFlags = 0;

            D3D11_DEPTH_STENCIL_VIEW_DESC dsVDesc;
            ZeroMemory(&dsVDesc, sizeof(D3D11_DEPTH_STENCIL_VIEW_DESC));     
            D3D_SAVE_CALL(g_pDevice->CreateTexture2D(&dsbDesc, NULL, &g_pDepthStencilBuffer));
            dsVDesc.Format = dsbDesc.Format;
            dsVDesc.ViewDimension = (g_samples > 1 ? D3D11_DSV_DIMENSION_TEXTURE2DMS : D3D11_DSV_DIMENSION_TEXTURE2D);
            D3D_SAVE_CALL(g_pDevice->CreateDepthStencilView(g_pDepthStencilBuffer, &dsVDesc, &g_pDepthStencilView));

            //m_pContext->OMSetRenderTargets(1, &m_pBackBufferView, m_pDepthStencilView);
            BindBackbuffer();

            SAFE_RELEASE(g_pBackBuffer);
        }

        VOID SetDefaultStates(VOID) 
        {
            g_pContext->PSSetSamplers(0, 4, g_pSamplerStates);
            g_pContext->VSSetSamplers(0, 4, g_pSamplerStates);
            //g_pContext->RSSetState(g_pRasterizerStateFrontFaceSolid);
            //g_pContext->OMSetDepthStencilState(m_pDepthWriteStencilState, 0);
            //g_pContext->OMSetBlendState(g_pBlendStateNoBlending, NULL, 0xffffff);
            BindBackbuffer();
            SetDefaultViewPort();

            g_pContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        }

        BOOL GetFullscreenState(VOID)
        {
            BOOL state;
            D3D_SAVE_CALL(g_pSwapChain->GetFullscreenState(&state, NULL));
            return state;
        }

        VOID SetFullscreenState(BOOL fs, UINT width, UINT height)
        {
            if(fs == GetFullscreenState())
            {
                return;
            }

            if(fs)
            {
                GetFullscreenSize(&width, &height);
            }

            DXGI_MODE_DESC desc;
            ZeroMemory(&desc, sizeof(DXGI_MODE_DESC));
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.Height = height;
            desc.Width = width;
            desc.RefreshRate.Denominator = 10000;
            desc.RefreshRate.Numerator = 60000;

            DisplayMode mode = desc;
            DisplayMode closest = GetClosestDisplayMode(mode);

            D3D_SAVE_CALL(g_pSwapChain->ResizeTarget(&closest.mode));

            D3D_SAVE_CALL(g_pSwapChain->SetFullscreenState(fs, NULL));

            closest.mode.RefreshRate.Denominator = 0;
            closest.mode.RefreshRate.Numerator = 0;

            D3D_SAVE_CALL(g_pSwapChain->ResizeTarget(&closest.mode));
        }

        VOID Resize(UINT width, UINT height)
        {
            if(g_pSwapChain) 
            {
                ReleaseBackbuffer();

                D3D_SAVE_CALL(g_pSwapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH));

                CreateBackbuffer(width, height);

                D3D11_VIEWPORT port;
                port.MinDepth = 0;
                port.MaxDepth = 1;
                port.TopLeftX = 0;
                port.TopLeftY = 0;
                port.Width = (FLOAT)width;
                port.Height = (FLOAT)height;

                UINT ports = 1;
                g_pContext->RSSetViewports(ports, &port);

                g_width = width;
                g_height = height;
            }
        }

        BOOL _CreateWindow(
            WNDPROC wndProc ,
            HINSTANCE hInstance, LPCWSTR title, UINT width, UINT height) 
        {

            RECT rec = {0,0,width,height};
            AdjustWindowRect(&rec, WS_OVERLAPPEDWINDOW, FALSE);

            WNDCLASSEX wcex;

            wcex.cbSize = sizeof(WNDCLASSEX);

            wcex.style = CS_HREDRAW | CS_VREDRAW;
            wcex.lpfnWndProc = wndProc;
            wcex.cbClsExtra = 0;
            wcex.cbWndExtra = 0;
            wcex.hInstance = hInstance;
            wcex.hIcon = 0;// LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
            wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
            wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
            wcex.lpszMenuName = 0; //MAKEINTRESOURCE(IDC_D3D);
            wcex.lpszClassName = L"D3DCLASS";
            wcex.hIconSm = 0;//LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));

            RegisterClassEx(&wcex);

            //hInst = hInstance; // Store instance handle in our global variable

            g_hWnd = CreateWindow(wcex.lpszClassName, title, WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT, 0, rec.right - rec.left, rec.bottom - rec.top, NULL, NULL, hInstance, NULL);

            if (!g_hWnd)
            {
                return FALSE;
            }

            ShowWindow(g_hWnd, 10);
            UpdateWindow(g_hWnd);

            return TRUE;
        }

        ID3D11Device* GetDevice(VOID) 
        { 
            return g_pDevice; 
        };

        ID3D11DeviceContext* GetContext(VOID)
        { 
            return g_pContext; 
        };

        std::string GetShaderError(ID3D10Blob* message) 
        {
            if(!message) return std::string();
            char* compileErrors = (char*)message->GetBufferPointer();
            std::string error;
            for(int i = 0; i < message->GetBufferSize(); ++i)
            {
                error += compileErrors[i];
            }
            message->Release();
            return error;
        }

        DXGI_FORMAT GetD3DFormatFromCMFormat(GraphicsFormat format)
        {
            return d3dFormatMap[format];
        }

        VOID Release(VOID) 
        {
            ID3D11Buffer* null[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT];
            TBD_FOR_INT(D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT)
            {
                null[i] = NULL;
            }
            chimera::d3d::GetContext()->VSSetConstantBuffers(0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, null);
            chimera::d3d::GetContext()->PSSetConstantBuffers(0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, null);
            chimera::d3d::GetContext()->GSSetConstantBuffers(0, D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT, null);

            chimera::d3d::GetContext()->IASetInputLayout(NULL);
            chimera::d3d::GetContext()->VSSetShader(NULL, NULL, 0);
            chimera::d3d::GetContext()->PSSetShader(NULL, NULL, 0);
            chimera::d3d::GetContext()->GSSetShader(NULL, NULL, 0);

            ID3D11SamplerState* ss[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];
            TBD_FOR_INT(D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT)
            {
                ss[i] = NULL;
            }
            g_pContext->PSSetSamplers(0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, ss);
            g_pContext->VSSetSamplers(0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, ss);

            g_pContext->RSSetState(NULL);
            g_pContext->OMSetDepthStencilState(NULL, 0);
            g_pContext->OMSetBlendState(NULL, NULL, 0xffffff);

            ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
            TBD_FOR_INT(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT) { rtv[i] = NULL; }
            chimera::d3d::GetContext()->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtv, NULL);

            ID3D11ShaderResourceView* srv[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT ];
            TBD_FOR_INT(D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT ) { srv[i] = NULL; }
            chimera::d3d::GetContext()->VSSetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT , srv);
            chimera::d3d::GetContext()->PSSetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT , srv);
            chimera::d3d::GetContext()->GSSetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT , srv);

            DestroyWindow(g_hWnd);

            TBD_FOR_INT(4) { g_pSamplerStates[i]->Release(); }
            SAFE_DELETE(g_adapterName);
            SAFE_RELEASE(g_pAdapter);
            SAFE_RELEASE(g_pDXGIFactory);
            SAFE_RELEASE(m_pNoDepthNoStencilState);
            SAFE_RELEASE(m_pDepthCmpStencilState);
            SAFE_RELEASE(m_pDepthWriteStencilState);
            SAFE_RELEASE(m_pDepthNoStencilState);
            SAFE_RELEASE(g_pBackBuffer);
            SAFE_RELEASE(g_pBlendStateNoBlending);
            SAFE_RELEASE(g_pBlendStateBlendAdd);
            SAFE_RELEASE(g_pBlendStateBlendAlpha);
            SAFE_RELEASE(g_pRasterizerStateFrontFaceSolid);
            SAFE_RELEASE(g_pRasterizerStateNoCullingSolid);
            SAFE_RELEASE(g_pRasterizerStateBackFaceSolid);
            SAFE_RELEASE(g_pRasterizerStateWrireframe);
            g_pSwapChain->SetFullscreenState(FALSE, NULL);
            SAFE_RELEASE(g_pSwapChain);
            ReleaseBackbuffer();
            SAFE_RELEASE(g_pContext);
            
            if(flags & D3D11_CREATE_DEVICE_DEBUG)
            {

                ID3D11Debug *debugDev;
                g_pDevice->QueryInterface(__uuidof(ID3D11Debug), (void**)&debugDev);
                debugDev->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);
                SAFE_RELEASE(debugDev);


                SAFE_RELEASE(g_pDevice);
                
/*
                typedef HRESULT(__stdcall *fPtr)(const IID&, void**); 
                HMODULE hDll = GetModuleHandleW(L"dxgidebug.dll"); 
                fPtr DXGIGetDebugInterface = (fPtr)GetProcAddress(hDll, "DXGIGetDebugInterface");
                IDXGIDebug* dxgidebug;
                DXGIGetDebugInterface(__uuidof(IDXGIDebug), (void**)&dxgidebug); 
                dxgidebug->ReportLiveObjects(DXGI_DEBUG_DXGI, DXGI_DEBUG_RLO_SUMMARY);
                SAFE_RELEASE(dxgidebug);
*/
            }
            else
            {
                SAFE_RELEASE(g_pDevice);
            }
        }
    }
}