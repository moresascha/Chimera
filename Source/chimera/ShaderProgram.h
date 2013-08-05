#pragma once
#include "stdafx.h"
#include "d3d.h"
#include "ConstBuffer.h"
#include <vector>
#include "Process.h"
#include "GameLogic.h"
namespace d3d 
{
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
            std::string path = app::g_pApp->GetConfig()->GetString("sShaderPath");
            std::wstring wpath = std::wstring(path.begin(), path.end()) + file;
            T* shader = new T(wpath.c_str(), function);

            ErrorLog log;
            if(!shader->Compile(&log))
            {
                LOG_CRITICAL_ERROR(log.c_str());
            }

#ifdef _DEBUG
            if(app::g_pApp->GetLogic())
            {
                std::shared_ptr<proc::WatchShaderFileModificationProcess> modProc = 
                    std::shared_ptr<proc::WatchShaderFileModificationProcess>(new proc::WatchShaderFileModificationProcess(shader, file, L"../Assets/shader/"));
                app::g_pApp->GetLogic()->AttachProcess(modProc);
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
    }

    class Shader
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
        virtual BOOL Compile(ErrorLog* errorLog) = 0;
        virtual VOID Bind(VOID) = 0;
        virtual VOID Unbind(VOID) = 0;
    };

    class PixelShader : public Shader
    {
    private:
        ID3D11PixelShader* m_pShader;
    public:
        PixelShader(LPCTSTR file, LPCSTR function);
        BOOL Compile(ErrorLog* errorLog);
        VOID Bind(VOID);
        VOID Unbind(VOID);
        
        static d3d::PixelShader* CreateShader(LPCSTR name, LPCTSTR file, LPCSTR function);
        static d3d::PixelShader* GetShader(LPCSTR name);
        static d3d::PixelShader* m_sCurrent;
        static std::map<std::string, PixelShader*>* m_spShader;
        static CONST std::map<std::string, PixelShader*>* GetShader(VOID) { return m_spShader; }
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
        BOOL Compile(ErrorLog* errorLog);
        VOID SetInputAttr(LPCSTR name, UINT position, UINT slot, UINT offset, DXGI_FORMAT format);
        VOID SetInputAttr(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format);
        VOID SetInputAttrInstanced(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format);
        VOID GenerateLayout(VOID);
        ID3D11InputLayout* GetInputLayout(VOID);
        VOID Bind(VOID);
        VOID Unbind(VOID);
        
        static d3d::VertexShader* CreateShader(LPCSTR name, LPCTSTR file, LPCSTR function);
        static d3d::VertexShader* GetShader(LPCSTR name);
        static d3d::VertexShader* m_sCurrent;
        static std::map<std::string, VertexShader*>* m_spShader;
        static CONST std::map<std::string, VertexShader*>* GetShader(VOID) { return m_spShader; }
        ~VertexShader(VOID);
    };

    class GeometryShader : public Shader
    {
    private:
        ID3D11GeometryShader* m_pShader;

    public:
        GeometryShader(LPCTSTR file, LPCSTR function);
        BOOL Compile(ErrorLog* errorLog);
        VOID Bind(VOID);
        VOID Unbind(VOID);
        
        static d3d::GeometryShader* CreateShader(LPCSTR name, LPCTSTR file, LPCSTR function);
        static d3d::GeometryShader* GetShader(LPCSTR name);
        static d3d::GeometryShader* m_sCurrent;
        static std::map<std::string, GeometryShader*>* m_spShader;
        static CONST std::map<std::string, GeometryShader*>* GetShader(VOID) { return m_spShader; }
        ~GeometryShader(VOID);
    };
    
    class ShaderProgram
    {
    private:
        PixelShader* m_pPixelShader;
        VertexShader* m_pVertexShader;
        GeometryShader* m_pGeometryShader;

        std::string m_functionVS;
        std::string m_functionPS;
        std::string m_functionGS;
        std::wstring m_file;
        std::string m_name;

        ShaderProgram(LPCSTR name, LPCTSTR file, LPCSTR functionVS, LPCSTR functionPS, LPCSTR functionGS = NULL);

        static std::map<std::string, std::shared_ptr<ShaderProgram>>* g_spPrograms;

    public:
        BOOL CompileShader(ErrorLog* errorLog = NULL);
        BOOL Bind(VOID);
        VOID Unbind(VOID);
        VOID GenerateLayout(VOID);
        VOID SetInputAttr(LPCSTR name, UINT position, UINT slot, UINT offset, DXGI_FORMAT format);
        VOID SetInputAttr(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format);
        VOID SetInputAttrInstanced(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format);

        /*LPCSTR GetPSFunction(VOID) CONST { return m_functionPS; }
        LPCSTR GetGSFunction(VOID) CONST { return m_functionGS; }
        LPCSTR GetVSFunction(VOID) CONST { return m_functionVS; } */
        LPCTSTR GetFile(VOID) CONST { return m_file.c_str(); }

        static VOID Create(VOID);
        static VOID Destroy(VOID);
        ~ShaderProgram(VOID);

        static std::shared_ptr<ShaderProgram> CreateProgram(LPCSTR name, LPCTSTR file, LPCSTR functionVS, LPCSTR functionPS, LPCSTR functionGS = NULL);
        static std::shared_ptr<ShaderProgram> GetProgram(LPCSTR name);
        static CONST std::map<std::string, std::shared_ptr<ShaderProgram>>* GetPrograms(VOID) { return g_spPrograms; }
    };
}
