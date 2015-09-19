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

        void PixelShader::VBind(void)
        {
            if(PixelShader::m_sCurrent == this)
            {
                return;
            }
            chimera::d3d::GetContext()->PSSetShader(m_pShader, NULL, 0);
            PixelShader::m_sCurrent = this;
        }

        void PixelShader::VUnbind(void)
        {
            chimera::d3d::GetContext()->PSSetShader(NULL, NULL, 0);
            PixelShader::m_sCurrent = NULL;
        }

        bool PixelShader::VCompile(ErrorLog* errorLog)
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
                return false;
            }
            else if(errorMessage != NULL)
            {
                if(errorLog)
                {
                    *errorLog = chimera::d3d::GetShaderError(errorMessage);
                }
                return false;
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
                return false;
            }

            SAFE_RELEASE(shaderCode);
            SAFE_RELEASE(m_pShader);

            m_pShader = pixelShader;

            SAFE_RELEASE(errorMessage);

            return true;
        }

        PixelShader::~PixelShader(void)
        {
            SAFE_RELEASE(m_pShader);
        }

        VertexShader::VertexShader(LPCTSTR file, LPCSTR function) 
            : Shader(function, file), m_pShader(NULL), m_pVertexShaderCode(NULL), m_pLayout(NULL), m_numInputElemens(0), m_numInstancedInputElemens(0)
        {

        }

        void VertexShader::VBind(void)
        {
            if(m_sCurrent == this)
            {
                return;
            }
            chimera::d3d::GetContext()->IASetInputLayout(m_pLayout);
            chimera::d3d::GetContext()->VSSetShader(m_pShader, NULL, 0);
            m_sCurrent = this;
        }

        void VertexShader::VUnbind(void)
        {
            chimera::d3d::GetContext()->IASetInputLayout(NULL);
            chimera::d3d::GetContext()->VSSetShader(NULL, NULL, 0);
            m_sCurrent = NULL;
        }

        bool VertexShader::VCompile(ErrorLog* errorLog)
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
                return false;
            }
            else if(errorMessage != NULL)
            {
                if(errorLog)
                {
                    *errorLog = chimera::d3d::GetShaderError(errorMessage);
                }
                return false;
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
                return false;
            }

            SAFE_RELEASE(m_pVertexShaderCode);
            SAFE_RELEASE(m_pShader);

            m_pVertexShaderCode = vsShaderCode;
            m_pShader = vertexShader;

            SAFE_RELEASE(errorMessage);

            return true;
        }

        void VertexShader::SetInputAttr(LPCSTR name, uint position, uint slot, uint offset, DXGI_FORMAT format)
        {
            ZeroMemory(&m_layouts[position], sizeof(D3D11_INPUT_ELEMENT_DESC));
            m_layouts[position].AlignedByteOffset = offset;
            m_layouts[position].Format = format;
            m_layouts[position].InputSlot = slot;
            m_layouts[position].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
            m_layouts[position].SemanticName = name;
            m_numInputElemens++;
        }

        void VertexShader::SetInputAttr(LPCSTR name, uint position, uint slot, DXGI_FORMAT format) 
        {
            ZeroMemory(&m_layouts[position], sizeof(D3D11_INPUT_ELEMENT_DESC));
            m_layouts[position].AlignedByteOffset = m_numInputElemens == 0 ? 0 : D3D11_APPEND_ALIGNED_ELEMENT;
            m_layouts[position].Format = format;
            m_layouts[position].InputSlot = slot;
            m_layouts[position].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
            m_layouts[position].SemanticName = name;
            m_numInputElemens++;
        }

        void VertexShader::SetInputAttrInstanced(LPCSTR name, uint position, uint slot, DXGI_FORMAT format) 
        {
            ZeroMemory(&m_layouts[position], sizeof(D3D11_INPUT_ELEMENT_DESC));
            m_layouts[position].AlignedByteOffset = m_numInstancedInputElemens == 0 ? 0 : D3D11_APPEND_ALIGNED_ELEMENT;
            m_layouts[position].Format = format;
            m_layouts[position].InputSlot = slot;
            m_layouts[position].InputSlotClass = D3D11_INPUT_PER_INSTANCE_DATA;
            m_layouts[position].SemanticName = name;
            m_layouts[position].InstanceDataStepRate = 1;
            m_numInstancedInputElemens++;
            m_numInputElemens++;
        }

        ID3D11InputLayout* VertexShader::GetInputLayout(void)
        {
            return m_pLayout;
        }

        void VertexShader::GenerateLayout(void) 
        {
            SAFE_RELEASE(m_pLayout);
            D3D_SAVE_CALL(chimera::d3d::GetDevice()->CreateInputLayout(
                m_layouts, m_numInputElemens, m_pVertexShaderCode->GetBufferPointer(), 
                m_pVertexShaderCode->GetBufferSize(), &m_pLayout));
        }

        VertexShader::~VertexShader(void)
        {
            SAFE_RELEASE(m_pShader);
            SAFE_RELEASE(m_pVertexShaderCode);
            SAFE_RELEASE(m_pLayout);
        }

        GeometryShader::GeometryShader(LPCTSTR file, LPCSTR function) : Shader(function, file), m_pShader(NULL)
        {

        }

        bool GeometryShader::VCompile(ErrorLog* errorLog)
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
                return false;
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
                return false;
            }

            SAFE_RELEASE(m_pShader);
            SAFE_RELEASE(errorMessage);

            m_pShader = geometryShader;

            return true;
        }

        void GeometryShader::VBind(void)
        {
            if(GeometryShader::m_sCurrent == this)
            {
                return;
            }
            chimera::d3d::GetContext()->GSSetShader(this->m_pShader, NULL, 0);
            GeometryShader::m_sCurrent = this;
        }

        void GeometryShader::VUnbind(void)
        {
            chimera::d3d::GetContext()->GSSetShader(NULL, NULL, 0);
            GeometryShader::m_sCurrent = NULL;
        }

        GeometryShader::~GeometryShader(void)
        {
            SAFE_RELEASE(m_pShader);
        }

        ShaderProgram::ShaderProgram(void) :
            m_pPixelShader(NULL),
            m_pVertexShader(NULL),
            m_pGeometryShader(NULL)
        {

        }

        bool ShaderProgram::VCompile(ErrorLog* errorLog /* = NULL */)
        {
            if(!m_pVertexShader->VCompile(errorLog))
            {
                return false;
            }

            if(m_pPixelShader)
            {
                if(!m_pPixelShader->VCompile(errorLog))
                {
                    return false;
                }
            }

            if(m_pGeometryShader)
            {

                if(!m_pGeometryShader->VCompile(errorLog))
                {
                    return false;
                }
            }

            return true;
        }

        void ShaderProgram::VBind(void) 
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

        void ShaderProgram::VUnbind(void) 
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

        void ShaderProgram::VAddShader(IShader* shader)
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

        ShaderProgram::~ShaderProgram(void) 
        {
            SAFE_DELETE(m_pPixelShader);
            SAFE_DELETE(m_pVertexShader);
            SAFE_DELETE(m_pGeometryShader);
        }
    }
}