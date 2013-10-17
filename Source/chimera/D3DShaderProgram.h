#pragma once
#include "stdafx.h"
#include "D3DGraphics.h"

namespace chimera
{
    namespace d3d
    {
        /*
    namespace helper
    {
        template <class T>
        T* CreateShader(LPCSTR name, LPCTSTR file, LPCSTR function)
        {
            auto it = T::m_spShader->find(name);
            if(it != T::m_spShader->end())
            {
                return it->second;
            }
            std::string path = chimera::g_pApp->GetConfig()->GetString("sShaderPath");
            std::wstring wpath = std::wstring(path.begin(), path.end()) + file;
            T* shader = new T(wpath.c_str(), function);

            ErrorLog log;
            if(!shader->Compile(&log))
            {
                LOG_CRITICAL_ERROR(log.c_str());
            }

#ifdef _DEBUG
            if(chimera::g_pApp->GetLogic())
            {
                std::shared_ptr<chimera::WatchShaderFileModificationProcess> modProc = 
                    std::shared_ptr<chimera::WatchShaderFileModificationProcess>(new chimera::WatchShaderFileModificationProcess(shader, file, L"../Assets/shader/"));
                chimera::g_pApp->GetLogic()->AttachProcess(modProc);
            }
#endif

            (*(T::m_spShader))[name] = shader;
            return shader;
        }

        template <class T>
        T* GetShader(LPCSTR name)
        {
            auto it = T::m_spShader->find(name);
            if(it == T::m_spShader->end())
            {
                LOG_CRITICAL_ERROR("unkown shader program");
                return NULL;
            }
            return it->second;
        }

        template <class T>
        VOID Create(VOID)
        {
            if(T::m_spShader == NULL)
            {
                T::m_spShader = new std::map<std::string, T*>();
            }
        }

        template <class T>
        VOID Destroy(VOID)
        {
            TBD_FOR((*(T::m_spShader)))
            {
                SAFE_DELETE(it->second);
            }
            SAFE_DELETE(T::m_spShader);
        }
    } */

    class Shader : public IShader
    {
    protected:
        std::string m_function;
        std::wstring m_file;

    public:
        Shader(LPCSTR function, LPCTSTR file) : m_function(function), m_file(file)
        {

        }

        LPCSTR GetFunctionFile(VOID)
        {
            return m_function.c_str();
        }

        LPCTSTR GetFile(VOID)
        {
            return m_file.c_str();
        }

        virtual BOOL VCompile(ErrorLog* errorLog) = 0;

        virtual VOID VBind(VOID) = 0;

        virtual VOID VUnbind(VOID) = 0;
    };

    class PixelShader : public Shader
    {
    private:
        ID3D11PixelShader* m_pShader;

    public:
        PixelShader(LPCTSTR file, LPCSTR function);

        BOOL VCompile(ErrorLog* errorLog);

        ShaderType VGetType(VOID) { return eShaderType_FragmentShader; }

        VOID VBind(VOID);

        VOID VUnbind(VOID);
        
        static PixelShader* m_sCurrent;

        ~PixelShader(VOID);
    };

    class VertexShader : public Shader
    {
    private:
        ID3DBlob* m_pVertexShaderCode; //needed for inputlayout
        ID3D11VertexShader* m_pShader;
        ID3D11InputLayout* m_pLayout;
        D3D11_INPUT_ELEMENT_DESC m_layouts[16];
        UINT m_numInputElemens;
        
    public:
        VertexShader(LPCTSTR file, LPCSTR function);

        BOOL VCompile(ErrorLog* errorLog);

        ShaderType VGetType(VOID) { return eShaderType_VertexShader; }

        VOID SetInputAttr(LPCSTR name, UINT position, UINT slot, UINT offset, DXGI_FORMAT format);

        VOID SetInputAttr(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format);

        VOID SetInputAttrInstanced(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format);

        VOID GenerateLayout(VOID);

        ID3D11InputLayout* GetInputLayout(VOID);

        VOID VBind(VOID);

        VOID VUnbind(VOID);
        
        static VertexShader* m_sCurrent;
        ~VertexShader(VOID);
    };

    class GeometryShader : public Shader
    {
    private:
        ID3D11GeometryShader* m_pShader;

    public:
        GeometryShader(LPCTSTR file, LPCSTR function);

        BOOL VCompile(ErrorLog* errorLog);

        VOID VBind(VOID);

        VOID VUnbind(VOID);

        ShaderType VGetType(VOID) { return eShaderType_GeometryShader; }

        static GeometryShader* m_sCurrent;
        ~GeometryShader(VOID);
    };
    
    class ShaderProgram : public IShaderProgram
    {
    private:
        PixelShader* m_pPixelShader;
        VertexShader* m_pVertexShader;
        GeometryShader* m_pGeometryShader;

    public:
        ShaderProgram(VOID);

        BOOL VCompile(ErrorLog* errorLog = NULL);

        VOID VBind(VOID);

        VOID VAddShader(IShader* shader);

        VOID VUnbind(VOID);

        VOID GenerateLayout(VOID);

        ~ShaderProgram(VOID);
    };
    }
}