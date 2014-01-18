#include "Profiling.h"
#include <sstream>

namespace chimera
{
    D3DQuery::D3DQuery(LPCSTR infoText, D3D11_QUERY query, void* pData, uint size) : Query(infoText), m_queryType(query), m_pQuery(NULL), m_pData(pData), m_dataSize(size)
    {
        D3D11_QUERY_DESC desc;
        desc.Query = query;
        desc.MiscFlags = 0;
        D3D_SAVE_CALL(chimera::GetDevice()->CreateQuery(&desc, &m_pQuery));
    }

    void D3DQuery::VStart(void)
    {
        chimera::g_pContext->Begin(m_pQuery);
    }

    void D3DQuery::VEnd(void)
    {
        chimera::g_pContext->End(m_pQuery);
        while(S_OK != chimera::GetContext()->GetData(m_pQuery, m_pData, m_dataSize, 0))
        {
            //Sleep(1);
        }
    }

    D3DQuery::~D3DQuery(void)
    {
        SAFE_RELEASE(m_pQuery);
    }

    D3DOcclusionQuery::D3DOcclusionQuery(void) : D3DQuery("Fragments generated", D3D11_QUERY_OCCLUSION, &m_data, 8)
    {

    }

    void D3DOcclusionQuery::GetResultAsString(std::string& result) const
    {
        std::stringstream ss;
        ss << m_data;
        result = ss.str();
    }

    D3DPipelineStatisticsQuery::D3DPipelineStatisticsQuery(void) : D3DQuery("Pipelinestatistics", D3D11_QUERY_PIPELINE_STATISTICS, &m_data, sizeof(D3D11_QUERY_DATA_PIPELINE_STATISTICS))
    {

    }

    void D3DPipelineStatisticsQuery::GetResultAsString(std::string& result) const
    {
        std::stringstream ss;
        ss << "Vertices Read: ";
        ss << m_data.IAVertices;
        ss << "\n";

        ss << "Primitives Read: ";
        ss << m_data.IAPrimitives;
        ss << "\n";

        ss << "Vertexshader invocations ";
        ss << m_data.VSInvocations;
        ss << "\n";

        ss << "Geometryshader invocations: ";
        ss << m_data.GSInvocations;
        ss << "\n";

        ss << "Geometryshader Primitives: ";
        ss << m_data.GSPrimitives;
        ss << "\n";

        ss << "Primitives send to Rasterizer: ";
        ss << m_data.CPrimitives;
        ss << "\n";

        ss << "Pixelshader invocations: ";
        ss << m_data.PSInvocations;

        result = ss.str();
    }

    D3DTimestampDisjointQuery::D3DTimestampDisjointQuery(void) : D3DQuery("TimestampDJ", D3D11_QUERY_TIMESTAMP_DISJOINT, &m_data, sizeof(D3D11_QUERY_DATA_TIMESTAMP_DISJOINT))
    {

    }

    void D3DTimestampDisjointQuery::GetResultAsString(std::string& result) const
    {
        //--
    }

    D3DTimestampQuery::D3DTimestampQuery(void) : D3DQuery("Timestamp", D3D11_QUERY_TIMESTAMP, &m_data, sizeof(UINT64))
    {

    }

    void D3DTimestampQuery::GetResultAsString(std::string& result) const
    {
        //--
    }

    D3DTimeDeltaQuery::D3DTimeDeltaQuery(void) : D3DQuery("Time", D3D11_QUERY_TIMESTAMP, &m_data, sizeof(UINT64))
    {
        m_pDisjointQuery = new D3DTimestampDisjointQuery();
        m_pTS0 = new D3DTimestampQuery();
        m_pTS1 = new D3DTimestampQuery();
    }

    void D3DTimeDeltaQuery::VStart(void)
    {
        m_pDisjointQuery->VStart();

        m_pTS0->VEnd();
    }

    void D3DTimeDeltaQuery::VEnd(void)
    {
        m_pTS1->VEnd();

        m_pDisjointQuery->VEnd();
    }

    void D3DTimeDeltaQuery::GetResultAsString(std::string& result) const
    {
        D3D10_QUERY_DATA_TIMESTAMP_DISJOINT* djd = (D3D10_QUERY_DATA_TIMESTAMP_DISJOINT *)m_pDisjointQuery->GetData();
        if(djd->Disjoint)
        {
            return;
        }
        UINT64 t0 = (*(UINT64*)m_pTS0->GetData());
        UINT64 t1 = (*(UINT64*)m_pTS1->GetData());
        std::stringstream ss;
        /*ss << "T0=";
        ss << t0;
        ss << ", T1=";
        ss << t1;
        ss << ", dt (ms)="; */
        ss << ((t1 - t0) / (float)djd->Frequency) * 1000.0f;
        result = ss.str();
        //DEBUG_OUT_A("%s\n", result.c_str());
    }

    D3DTimeDeltaQuery::~D3DTimeDeltaQuery(void)
    {
        SAFE_DELETE(m_pDisjointQuery);
        SAFE_DELETE(m_pTS0);
        SAFE_DELETE(m_pTS1);
    }
}
