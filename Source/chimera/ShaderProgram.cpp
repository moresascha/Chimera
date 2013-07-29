#include "stdafx.h"
#include "ShaderProgram.h"
#include "Process.h"
#include "GameApp.h"

namespace d3d 
{
    std::map<std::string, std::shared_ptr<ShaderProgram>>* ShaderProgram::g_spPrograms = NULL;

    d3d::VertexShader* d3d::VertexShader::m_sCurrent = NULL;
    d3d::PixelShader* d3d::PixelShader::m_sCurrent = NULL;
    d3d::GeometryShader* d3d::GeometryShader::m_sCurrent = NULL;

    std::map<std::string, VertexShader*>* d3d::VertexShader::m_spShader;
    std::map<std::string, PixelShader*>* d3d::PixelShader::m_spShader;
    std::map<std::string, GeometryShader*>* d3d::GeometryShader::m_spShader;

    PixelShader::PixelShader(LPCTSTR file, LPCSTR function) : Shader(function, file), m_pShader(NULL)
    {

    }

    VOID PixelShader::Bind(VOID)
    {
        if(PixelShader::m_sCurrent == this)
        {
            return;
        }
        d3d::GetContext()->PSSetShader(this->m_pShader, NULL, 0);
        PixelShader::m_sCurrent = this;
    }

    VOID PixelShader::Unbind(VOID)
    {
        d3d::GetContext()->PSSetShader(NULL, NULL, 0);
        PixelShader::m_sCurrent = NULL;
    }

    BOOL PixelShader::Compile(ErrorLog* errorLog)
    {
        ID3D11PixelShader* pixelShader = NULL;
        ID3DBlob* errorMessage = NULL;
        ID3DBlob* shaderCode = NULL;

        HRESULT hr = D3DX11CompileFromFileW(m_file.c_str(), 0, 0, m_function.c_str(), d3d::g_pixelShaderMaxProfile, 0, 0, 0, &shaderCode, &errorMessage, 0);

        if(FAILED(hr) && errorMessage == NULL)
        {
            if(errorLog)
            {
                std::wstring sf(m_file);
                std::string s(sf.begin(), sf.end());
                *errorLog = "File not found: " + s;
            }
            return FALSE;
        }
        else if(errorMessage != NULL)
        {
            if(errorLog)
            {
                *errorLog = d3d::GetShaderError(errorMessage);
            }
            return FALSE;
        }

        hr = d3d::GetDevice()->CreatePixelShader(shaderCode->GetBufferPointer(), shaderCode->GetBufferSize(), NULL, &pixelShader);

        if(FAILED(hr))
        {
            SAFE_RELEASE(pixelShader);
            SAFE_RELEASE(shaderCode);
            if(errorLog)
            {
                *errorLog = "Failed to create Pixelshader";
            }
            return FALSE;
        }

        m_pShader = pixelShader;

        SAFE_RELEASE(errorMessage);

        return TRUE;
    }

    /*static*/ d3d::PixelShader* PixelShader::CreateShader(LPCSTR name, LPCTSTR file, LPCSTR function)
    {
        return helper::CreateShader<d3d::PixelShader>(name, file, function);
    }

    /*static*/ d3d::PixelShader* PixelShader::GetShader(LPCSTR name)
    {
        return helper::GetShader<d3d::PixelShader>(name);
    }

    PixelShader::~PixelShader(VOID)
    {
        SAFE_RELEASE(m_pShader);
    }

    VertexShader::VertexShader(LPCTSTR file, LPCSTR function) : Shader(function, file), m_pShader(NULL), m_pVertexShaderCode(NULL), m_pLayout(NULL), m_numInputElemens(0)
    {

    }

    VOID VertexShader::Bind(VOID)
    {
        if(m_sCurrent == this)
        {
            return;
        }
        d3d::GetContext()->IASetInputLayout(this->m_pLayout);
        d3d::GetContext()->VSSetShader(this->m_pShader, NULL, 0);
        m_sCurrent = this;
    }

    VOID VertexShader::Unbind(VOID)
    {
        d3d::GetContext()->IASetInputLayout(NULL);
        d3d::GetContext()->VSSetShader(NULL, NULL, 0);
        m_sCurrent = NULL;
    }

    BOOL VertexShader::Compile(ErrorLog* errorLog)
    {
        ID3D11VertexShader* vertexShader = NULL;
        ID3DBlob* errorMessage = NULL;
        ID3DBlob* vsShaderCode = NULL;

        HRESULT hr = D3DX11CompileFromFileW(m_file.c_str(), 0, 0, m_function.c_str(), d3d::g_vertexShaderMaxProfile, 0, 0, 0, &vsShaderCode, &errorMessage, 0);

        if(FAILED(hr) && errorMessage == NULL)
        {
            if(errorLog)
            {
                std::wstring sf(m_file);
                std::string s(sf.begin(), sf.end());
                *errorLog = "File not found: " + s;
            }
            return FALSE;
        }
        else if(errorMessage != NULL)
        {
            if(errorLog)
            {
                *errorLog = d3d::GetShaderError(errorMessage);
            }
            return FALSE;
        }

        hr = d3d::GetDevice()->CreateVertexShader(vsShaderCode->GetBufferPointer(), vsShaderCode->GetBufferSize(), NULL, &vertexShader);

        if(FAILED(hr))
        {
            SAFE_RELEASE(vertexShader);
            SAFE_RELEASE(vsShaderCode);
            if(errorLog)
            {
                *errorLog = "Failed to create Vertexshader";
            }
            return FALSE;
        }

        SAFE_RELEASE(m_pVertexShaderCode);
        SAFE_RELEASE(m_pShader);

        m_pVertexShaderCode = vsShaderCode;
        m_pShader = vertexShader;

        SAFE_RELEASE(errorMessage);

        return TRUE;
    }

    VOID VertexShader::SetInputAttr(LPCSTR name, UINT position, UINT slot, UINT offset, DXGI_FORMAT format)
    {
        ZeroMemory(&this->m_layouts[position], sizeof(D3D11_INPUT_ELEMENT_DESC));
        this->m_layouts[position].AlignedByteOffset = offset;
        this->m_layouts[position].Format = format;
        this->m_layouts[position].InputSlot = slot;
        this->m_layouts[position].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
        this->m_layouts[position].SemanticName = name;
        this->m_numInputElemens++;
    }

    VOID VertexShader::SetInputAttr(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format) 
    {
        ZeroMemory(&this->m_layouts[position], sizeof(D3D11_INPUT_ELEMENT_DESC));
        this->m_layouts[position].AlignedByteOffset = this->m_numInputElemens == 0 ? 0 : D3D11_APPEND_ALIGNED_ELEMENT;
        this->m_layouts[position].Format = format;
        this->m_layouts[position].InputSlot = slot;
        this->m_layouts[position].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
        this->m_layouts[position].SemanticName = name;
        this->m_numInputElemens++;
    }

    VOID VertexShader::SetInputAttrInstanced(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format) 
    {
        ZeroMemory(&this->m_layouts[position], sizeof(D3D11_INPUT_ELEMENT_DESC));
        this->m_layouts[position].AlignedByteOffset = 0; //TODO allow more attributes per instance
        this->m_layouts[position].Format = format;
        this->m_layouts[position].InputSlot = slot;
        this->m_layouts[position].InputSlotClass = D3D11_INPUT_PER_INSTANCE_DATA;
        this->m_layouts[position].SemanticName = name;
        this->m_layouts[position].InstanceDataStepRate = 1;
        this->m_numInputElemens++;
    }

    ID3D11InputLayout* VertexShader::GetInputLayout(VOID)
    {
        return m_pLayout;
    }

    VOID VertexShader::GenerateLayout(VOID) 
    {
        SAFE_RELEASE(m_pLayout);
        CHECK__(d3d::GetDevice()->CreateInputLayout(m_layouts, m_numInputElemens, m_pVertexShaderCode->GetBufferPointer(), m_pVertexShaderCode->GetBufferSize(), &this->m_pLayout));
    }

    /*static*/ d3d::VertexShader* VertexShader::CreateShader(LPCSTR name, LPCTSTR file, LPCSTR function)
    {
        return helper::CreateShader<d3d::VertexShader>(name, file, function);
    }

    d3d::VertexShader* VertexShader::GetShader(LPCSTR name)
    {
        return helper::GetShader<d3d::VertexShader>(name);
    }

    VertexShader::~VertexShader(VOID)
    {
        SAFE_RELEASE(m_pShader);
        SAFE_RELEASE(m_pVertexShaderCode);
        SAFE_RELEASE(m_pLayout);
    }

    GeometryShader::GeometryShader(LPCTSTR file, LPCSTR function) : Shader(function, file), m_pShader(NULL)
    {

    }

    BOOL GeometryShader::Compile(ErrorLog* errorLog)
    {
        ID3D11GeometryShader* geometryShader = NULL;
        ID3DBlob* errorMessage = NULL;
        ID3DBlob* shaderCode = NULL;

        HRESULT hr = D3DX11CompileFromFileW(m_file.c_str(), 0, 0, m_function.c_str(), d3d::g_geometryShaderMaxProfile, 0, 0, 0, &shaderCode, &errorMessage, 0);

        if(FAILED(hr))
        {
            if(errorLog)
            {
                *errorLog = d3d::GetShaderError(errorMessage);
            }
            return FALSE;
        }

        hr = d3d::GetDevice()->CreateGeometryShader(shaderCode->GetBufferPointer(), shaderCode->GetBufferSize(), NULL, &geometryShader);

        if(FAILED(hr))
        {
            SAFE_RELEASE(geometryShader);
            SAFE_RELEASE(shaderCode);
            if(errorLog)
            {
                *errorLog = "Failed to create GeometryShader";
            }
            return FALSE;
        }

        SAFE_RELEASE(m_pShader);
        SAFE_RELEASE(errorMessage);

        m_pShader = geometryShader;
        
        return TRUE;
    }

    VOID GeometryShader::Bind(VOID)
    {
        if(GeometryShader::m_sCurrent == this)
        {
            return;
        }
        d3d::GetContext()->GSSetShader(this->m_pShader, NULL, 0);
        GeometryShader::m_sCurrent = this;
    }

    VOID GeometryShader::Unbind(VOID)
    {
        d3d::GetContext()->GSSetShader(NULL, NULL, 0);
        GeometryShader::m_sCurrent = NULL;
    }

    /*static*/ d3d::GeometryShader* GeometryShader::CreateShader(LPCSTR name, LPCTSTR file, LPCSTR function)
    {
        return helper::CreateShader<d3d::GeometryShader>(name, file, function);
    }

    d3d::GeometryShader* GeometryShader::GetShader(LPCSTR name)
    {
        return helper::GetShader<d3d::GeometryShader>(name);
    }

    GeometryShader::~GeometryShader(VOID)
    {
        SAFE_RELEASE(m_pShader);
    }

    ShaderProgram::ShaderProgram(LPCSTR name, LPCTSTR file, LPCSTR functionVS, LPCSTR functionPS, LPCSTR functionGS) : 
        m_pPixelShader(NULL),
        m_pVertexShader(NULL),
        m_pGeometryShader(NULL),
        m_file(file), 
        m_name(name)
    {
        if(functionVS)
        {
            m_functionVS = functionVS;
        }
        else
        {
            m_functionVS = "error";
        }
        
        if(functionGS)
        {
            m_functionGS = functionGS;
        }
        else
        {
            m_functionGS = "";
        }
        if(functionPS)
        {
            m_functionPS = functionPS;
        }
        else
        {
            m_functionPS = "";
        }
    }

    BOOL ShaderProgram::CompileShader(ErrorLog* errorLog /* = NULL */)
    {
        PixelShader* pixelShader = NULL;
        VertexShader* vertexShader = NULL;
        GeometryShader* geometryShader = NULL;

        vertexShader = d3d::VertexShader::CreateShader(m_name.c_str(), m_file.c_str(), m_functionVS.c_str());

        if(!vertexShader->Compile(errorLog))
        {
            //SAFE_DELETE(vertexShader);
            return FALSE;
        }

        if(m_functionPS != "")
        {
            pixelShader = d3d::PixelShader::CreateShader(m_name.c_str(), m_file.c_str(), m_functionPS.c_str());

            if(!pixelShader->Compile(errorLog))
            {
                //SAFE_DELETE(pixelShader);
                return FALSE;
            }
        }

        if(m_functionGS != "")
        {
            geometryShader = d3d::GeometryShader::CreateShader(m_name.c_str(), m_file.c_str(), m_functionGS.c_str());

            if(!geometryShader->Compile(errorLog))
            {
                //SAFE_DELETE(geometryShader);
                return FALSE;
            }
        }


        //if we got here we can set the new program
        SAFE_DELETE(m_pGeometryShader);
        SAFE_DELETE(m_pPixelShader);
        SAFE_DELETE(m_pVertexShader);

        m_pPixelShader = pixelShader;
        m_pVertexShader = vertexShader;
        m_pGeometryShader = geometryShader;

        return TRUE;
    }

    BOOL ShaderProgram::Bind(VOID) 
    {
        m_pVertexShader->Bind();
        m_pPixelShader->Bind();

        if(m_pGeometryShader)
        {
            m_pGeometryShader->Bind();
        }
        else
        {
            d3d::GetContext()->GSSetShader(NULL, NULL, 0);
        }

        return TRUE;
    }

    VOID ShaderProgram::Unbind(VOID) 
    {
        m_pVertexShader->Bind();
        m_pPixelShader->Bind();
        m_pGeometryShader->Bind();
    }

    VOID ShaderProgram::SetInputAttr(LPCSTR name, UINT position, UINT slot, UINT offset, DXGI_FORMAT format)
    {
        m_pVertexShader->SetInputAttr(name, position, slot, offset, format);
    }

    VOID ShaderProgram::SetInputAttr(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format) 
    {
        m_pVertexShader->SetInputAttr(name, position, slot, format);
    }

    VOID ShaderProgram::SetInputAttrInstanced(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format) 
    {
        m_pVertexShader->SetInputAttrInstanced(name, position, slot, format);
    }

    VOID ShaderProgram::GenerateLayout(VOID) 
    {
        m_pVertexShader->GenerateLayout();
    }

    ShaderProgram::~ShaderProgram(VOID) 
    {
    }

    std::shared_ptr<ShaderProgram> ShaderProgram::GetProgram(LPCSTR name)
    {
        auto it = g_spPrograms->find(name);
#ifdef _DEBUG
        if(it == g_spPrograms->end())
        {
            LOG_CRITICAL_ERROR((std::string("Shaderprogram '") + name + std::string("' does not exist!")).c_str());
            return NULL;
        }
#endif
        return it->second;
    }

    std::shared_ptr<ShaderProgram> ShaderProgram::CreateProgram(LPCSTR name, LPCTSTR file, LPCSTR functionVS, LPCSTR functionPS, LPCSTR functionGS)
    {
        auto it = g_spPrograms->find(name);
        if(it != g_spPrograms->end())
        {
            return it->second;
        }

        std::shared_ptr<d3d::ShaderProgram> prog = std::shared_ptr<d3d::ShaderProgram>(new d3d::ShaderProgram(name, file, functionVS, functionPS, functionGS));

        ErrorLog log;
        if(!prog->CompileShader(&log))
        {
            LOG_CRITICAL_ERROR(log.c_str());
        }

        (*g_spPrograms)[name] = prog;

        return prog;
    }

    VOID ShaderProgram::Create(VOID)
    {
        if(g_spPrograms == NULL)
        {
            g_spPrograms = new std::map<std::string, std::shared_ptr<ShaderProgram>>();
        }
        helper::Create<d3d::VertexShader>();
        helper::Create<d3d::PixelShader>();
        helper::Create<d3d::GeometryShader>();
    }

    VOID ShaderProgram::Destroy(VOID)
    {
        helper::Destroy<d3d::VertexShader>();
        helper::Destroy<d3d::PixelShader>();
        helper::Destroy<d3d::GeometryShader>();
        SAFE_DELETE(g_spPrograms);
    }
}
