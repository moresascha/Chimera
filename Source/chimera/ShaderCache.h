#pragma once
#include "stdafx.h"

namespace chimera
{
    typedef std::map<std::string, IShader*> ShaderMap;
    typedef std::map<std::string, IShaderProgram*> ShaderProgramMap;

    #define NYC_ERROR LOG_CRITICAL_ERROR_A("ShaderProgram/Shader '%s' not yet created!", name)

    class ShaderCache : public IShaderCache
    {
    private:
        ShaderMap m_shaderMap;
        ShaderProgramMap m_programMap;
        std::unique_ptr<IShaderFactory> m_pFactory;

        template <typename T>
        IShader* GetShader(LPCSTR name, T t)
        {
            auto it = m_shaderMap.find(name);
            if(it != m_shaderMap.end())
            {
                return it->second;
            }

            NYC_ERROR;

            return NULL;
        }

    public:

        ShaderCache(std::unique_ptr<IShaderFactory> factory) : m_pFactory(std::move(factory))
        {

        }

        IShader* VGetVertexShader(LPCSTR name)
        {
            return GetShader(name, eShaderType_VertexShader);
        }

        IShader* VGetFragmentShader(LPCSTR name)
        {
            return GetShader(name, eShaderType_VertexShader);
        }

        IShader* VGetGeometryShader(LPCSTR name)
        {
            return GetShader(name, eShaderType_VertexShader);
        }

        IShaderProgram* VGetShaderProgram(LPCSTR name)
        {
            auto it = m_programMap.find(name);

            if(it != m_programMap.end())
            {
                return it->second;
            }
            
            NYC_ERROR;

            return NULL;
        }

        IShader* VCreateShader(LPCSTR name, CONST CMShaderDescription* desc, ShaderType t)
        {
            auto it = m_shaderMap.find(name);

            if(it != m_shaderMap.end())
            {
                return it->second;
            }

            IShader* shader;

            if(t == eShaderType_VertexShader)
            {
                shader = m_pFactory->VCreateVertexShader((CMVertexShaderDescription*)desc);
            }
            else
            if(t == eShaderType_FragmentShader)
            {
                shader = m_pFactory->VCreateFragmentShader(desc);
            }
            else
            if(t == eShaderType_GeometryShader)
            {
                shader = m_pFactory->VCreateGeometryShader(desc);
            }

            m_shaderMap[name] = shader;

            return shader;
        }

        IShaderProgram* VCreateShaderProgram(LPCSTR name, CONST CMShaderProgramDescription* desc)
        {
            auto it = m_programMap.find(name);

            if(it != m_programMap.end())
            {
                return it->second;
            }

            IShaderProgram* p = m_pFactory->VCreateShaderProgram(desc);

            m_programMap[name] = p;

            ErrorLog log;
            if(!p->VCompile(&log))
            {
                LOG_CRITICAL_ERROR_A("%s", log.c_str());
            }

            return p;
        }

        ~ShaderCache(VOID)
        {
            TBD_FOR(m_programMap)
            {
                SAFE_DELETE(it->second);
            }

            TBD_FOR(m_shaderMap)
            {
                SAFE_DELETE(it->second);
            }
        }
    };
}