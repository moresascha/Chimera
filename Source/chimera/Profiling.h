#pragma once
#include "stdafx.h"
#include "d3d.h"

namespace tbd
{
    class Query
    {
    private:
        std::string m_infoText;
    public:
        Query(LPCSTR infoText) : m_infoText(infoText) {}
        virtual VOID VStart(VOID) = 0;
        virtual VOID VEnd(VOID) = 0;
        virtual VOID GetResultAsString(std::string& result) CONST = 0;
        LPCSTR GetInfo(VOID) CONST { return m_infoText.c_str(); }
        virtual ~Query(VOID) {}
    };

    class D3DQuery : public Query
    {
    private:
        D3D11_QUERY m_queryType;
        ID3D11Query* m_pQuery;
        VOID* m_pData;
        UINT m_dataSize;
    public:
        D3DQuery(LPCSTR infoText, D3D11_QUERY query, VOID* pData, UINT size);
        virtual VOID VStart(VOID);
        virtual VOID VEnd(VOID);
        VOID* GetData(VOID) { return m_pData; }
        virtual ~D3DQuery(VOID);
    };

    class D3DOcclusionQuery : public D3DQuery
    {
    private:
        UINT64 m_data;
    public:
        D3DOcclusionQuery(VOID);
        VOID GetResultAsString(std::string& result) CONST;
        ~D3DOcclusionQuery(VOID) {}
    };

    class D3DPipelineStatisticsQuery : public D3DQuery
    {
    private:
        D3D11_QUERY_DATA_PIPELINE_STATISTICS m_data;
    public:
        D3DPipelineStatisticsQuery(VOID);
        VOID GetResultAsString(std::string& result) CONST;
        ~D3DPipelineStatisticsQuery(VOID) {}
    };

    class D3DTimestampDisjointQuery : public D3DQuery
    {
    private:
        D3D11_QUERY_DATA_TIMESTAMP_DISJOINT m_data;
    public:
        D3DTimestampDisjointQuery(VOID);
        VOID GetResultAsString(std::string& result) CONST;
        ~D3DTimestampDisjointQuery(VOID) {}
    };

    class D3DTimestampQuery : public D3DQuery
    {
    private:
        UINT64 m_data;
    public:
        D3DTimestampQuery(VOID);
        VOID GetResultAsString(std::string& result) CONST;
        ~D3DTimestampQuery(VOID){}
    };

    class D3DTimeDeltaQuery : public D3DQuery
    {
    private:
        UINT64 m_data;
        D3DTimestampDisjointQuery* m_pDisjointQuery;
        D3DTimestampQuery* m_pTS0;
        D3DTimestampQuery* m_pTS1;
    public:
        D3DTimeDeltaQuery(VOID);
        VOID GetResultAsString(std::string& result) CONST;
        VOID VStart(VOID);
        VOID VEnd(VOID);
        ~D3DTimeDeltaQuery(VOID);
    };
}