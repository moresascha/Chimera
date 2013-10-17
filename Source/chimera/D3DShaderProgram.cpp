#include "D3DShaderProgram.h"
#include <D3Dcompiler.h>
namespace chimera
{
    namespace d3d
    {
        GeometryShader* GeometryShader::m_sCurrent = NULL;
        VertexShader* VertexShader::m_sCurrent = NULL;
        PixelShader* PixelShader::m_sCurrent = NULL;

        PixelShader::PixelShader(LPCTSTR file, LPCSTR function) : Shader(function, file), m_pShader(NULL)
        {

        }

        VOID PixelShader::VBind(VOID)
        {
            if(PixelShader::m_sCurrent == this)
            {
                return;
            }
            chimera::d3d::GetContext()->PSSetShader(this->m_pShader, NULL, 0);
            PixelShader::m_sCurrent = this;
        }

        VOID PixelShader::VUnbind(VOID)
        {
            chimera::d3d::GetContext()->PSSetShader(NULL, NULL, 0);
            PixelShader::m_sCurrent = NULL;
        }

        BOOL PixelShader::VCompile(ErrorLog* errorLog)
        {
            ID3D11PixelShader* pixelShader = NULL;
            ID3DBlob* errorMessage = NULL;
            ID3DBlob* shaderCode = NULL;

            HRESULT hr = D3DCompileFromFile(m_file.c_str(), 0, D3D_COMPILE_STANDARD_FILE_INCLUDE, m_function.c_str(), d3d::g_pixelShaderMaxProfile, 0, 0, &shaderCode, &errorMessage);

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
                    *errorLog = chimera::d3d::GetShaderError(errorMessage);
                }
                return FALSE;
            }

            hr = chimera::d3d::GetDevice()->CreatePixelShader(shaderCode->GetBufferPointer(), shaderCode->GetBufferSize(), NULL, &pixelShader);

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

            SAFE_RELEASE(shaderCode);
            SAFE_RELEASE(m_pShader);

            m_pShader = pixelShader;

            SAFE_RELEASE(errorMessage);

            return TRUE;
        }

        PixelShader::~PixelShader(VOID)
        {
            SAFE_RELEASE(m_pShader);
        }

        VertexShader::VertexShader(LPCTSTR file, LPCSTR function) : Shader(function, file), m_pShader(NULL), m_pVertexShaderCode(NULL), m_pLayout(NULL), m_numInputElemens(0)
        {

        }

        VOID VertexShader::VBind(VOID)
        {
            if(m_sCurrent == this)
            {
                return;
            }
            chimera::d3d::GetContext()->IASetInputLayout(m_pLayout);
            chimera::d3d::GetContext()->VSSetShader(m_pShader, NULL, 0);
            m_sCurrent = this;
        }

        VOID VertexShader::VUnbind(VOID)
        {
            chimera::d3d::GetContext()->IASetInputLayout(NULL);
            chimera::d3d::GetContext()->VSSetShader(NULL, NULL, 0);
            m_sCurrent = NULL;
        }

        BOOL VertexShader::VCompile(ErrorLog* errorLog)
        {
            ID3D11VertexShader* vertexShader = NULL;
            ID3DBlob* errorMessage = NULL;
            ID3DBlob* vsShaderCode = NULL;

            HRESULT hr = D3DCompileFromFile(m_file.c_str(), 0, D3D_COMPILE_STANDARD_FILE_INCLUDE, m_function.c_str(), g_vertexShaderMaxProfile, 0, 0, &vsShaderCode, &errorMessage);

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
                    *errorLog = chimera::d3d::GetShaderError(errorMessage);
                }
                return FALSE;
            }

            hr = chimera::d3d::GetDevice()->CreateVertexShader(vsShaderCode->GetBufferPointer(), vsShaderCode->GetBufferSize(), NULL, &vertexShader);

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
            ZeroMemory(&m_layouts[position], sizeof(D3D11_INPUT_ELEMENT_DESC));
            m_layouts[position].AlignedByteOffset = offset;
            m_layouts[position].Format = format;
            m_layouts[position].InputSlot = slot;
            m_layouts[position].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
            m_layouts[position].SemanticName = name;
            m_numInputElemens++;
        }

        VOID VertexShader::SetInputAttr(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format) 
        {
            ZeroMemory(&m_layouts[position], sizeof(D3D11_INPUT_ELEMENT_DESC));
            m_layouts[position].AlignedByteOffset = m_numInputElemens == 0 ? 0 : D3D11_APPEND_ALIGNED_ELEMENT;
            m_layouts[position].Format = format;
            m_layouts[position].InputSlot = slot;
            m_layouts[position].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
            m_layouts[position].SemanticName = name;
            m_numInputElemens++;
        }

        VOID VertexShader::SetInputAttrInstanced(LPCSTR name, UINT position, UINT slot, DXGI_FORMAT format) 
        {
            ZeroMemory(&this->m_layouts[position], sizeof(D3D11_INPUT_ELEMENT_DESC));
            m_layouts[position].AlignedByteOffset = 0; //TODO allow more attributes per instance
            m_layouts[position].Format = format;
            m_layouts[position].InputSlot = slot;
            m_layouts[position].InputSlotClass = D3D11_INPUT_PER_INSTANCE_DATA;
            m_layouts[position].SemanticName = name;
            m_layouts[position].InstanceDataStepRate = 1;
            m_numInputElemens++;
        }

        ID3D11InputLayout* VertexShader::GetInputLayout(VOID)
        {
            return m_pLayout;
        }

        VOID VertexShader::GenerateLayout(VOID) 
        {
            SAFE_RELEASE(m_pLayout);
            D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateInputLayout(m_layouts, m_numInputElemens, m_pVertexShaderCode->GetBufferPointer(), m_pVertexShaderCode->GetBufferSize(), &this->m_pLayout));
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

        BOOL GeometryShader::VCompile(ErrorLog* errorLog)
        {
            ID3D11GeometryShader* geometryShader = NULL;
            ID3DBlob* errorMessage = NULL;
            ID3DBlob* shaderCode = NULL;

            HRESULT hr = D3DCompileFromFile(m_file.c_str(), 0, D3D_COMPILE_STANDARD_FILE_INCLUDE, m_function.c_str(), g_geometryShaderMaxProfile, 0, 0, &shaderCode, &errorMessage);

            if(FAILED(hr))
            {
                if(errorLog)
                {
                    *errorLog = chimera::d3d::GetShaderError(errorMessage);
                }
                return FALSE;
            }

            hr = chimera::d3d::GetDevice()->CreateGeometryShader(shaderCode->GetBufferPointer(), shaderCode->GetBufferSize(), NULL, &geometryShader);

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

        VOID GeometryShader::VBind(VOID)
        {
            if(GeometryShader::m_sCurrent == this)
            {
                return;
            }
            chimera::d3d::GetContext()->GSSetShader(this->m_pShader, NULL, 0);
            GeometryShader::m_sCurrent = this;
        }

        VOID GeometryShader::VUnbind(VOID)
        {
            chimera::d3d::GetContext()->GSSetShader(NULL, NULL, 0);
            GeometryShader::m_sCurrent = NULL;
        }

        GeometryShader::~GeometryShader(VOID)
        {
            SAFE_RELEASE(m_pShader);
        }

        ShaderProgram::ShaderProgram(VOID) :
            m_pPixelShader(NULL),
            m_pVertexShader(NULL),
            m_pGeometryShader(NULL)
        {

        }

        BOOL ShaderProgram::VCompile(ErrorLog* errorLog /* = NULL */)
        {
            if(!m_pVertexShader->VCompile(errorLog))
            {
                return FALSE;
            }

            if(m_pPixelShader)
            {
                if(!m_pPixelShader->VCompile(errorLog))
                {
                    return FALSE;
                }
            }

            if(m_pGeometryShader)
            {

                if(!m_pGeometryShader->VCompile(errorLog))
                {
                    return FALSE;
                }
            }

            return TRUE;
        }

        VOID ShaderProgram::VBind(VOID) 
        {
            m_pVertexShader->VBind();
            m_pPixelShader->VBind();

            if(m_pGeometryShader)
            {
                m_pGeometryShader->VBind();
            }
            else
            {
                chimera::d3d::GetContext()->GSSetShader(NULL, NULL, 0);
            }
        }

        VOID ShaderProgram::VUnbind(VOID) 
        {
            m_pVertexShader->VBind();
            if(m_pPixelShader)
            {
                m_pPixelShader->VBind();
            }
            if(m_pGeometryShader)
            {
                m_pGeometryShader->VBind();
            }
        }

        VOID ShaderProgram::VAddShader(IShader* shader)
        {
            if(shader->VGetType() == eShaderType_FragmentShader)
            {
                m_pPixelShader = (PixelShader*)shader;
            }
            else if(shader->VGetType() == eShaderType_VertexShader)
            {
                m_pVertexShader = (VertexShader*)shader;
            }
            else if(shader->VGetType() == eShaderType_GeometryShader)
            {
                m_pGeometryShader = (GeometryShader*)shader;
            }
        }

        ShaderProgram::~ShaderProgram(VOID) 
        {
            SAFE_DELETE(m_pPixelShader);
            SAFE_DELETE(m_pVertexShader);
            SAFE_DELETE(m_pGeometryShader);
        }
    }
}