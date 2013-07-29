#include "Config.h"
#include <fstream>
#include "util.h"
namespace util
{

#define CHECK_VECTOR_SIZE(vector) \
    if((vector).size() != 2) \
    { \
        LOG_ERROR_A("Error parsing config.ini: %s", line.c_str()); \
    } \

    typedef std::vector<std::string> NameValuePair;

    BOOL CheckValue(std::string& valueLine)
    {
        return valueLine.find("=") != std::string::npos;
    }

    template<class T>
    BOOL GetNumericValue(T& t, std::string& value)
    {
        std::stringstream ss;
        ss << value;
        ss >> t;
        return !ss.fail();
    }

    BOOL Config::LoadConfigFile(LPCSTR file)
    {
        std::ifstream stream(file);

        std::string line;

        while(stream.good())
        {
            std::getline(stream, line);
            if(line.size() > 0)
            {
                if(line[0] == '[')
                {
                    continue;
                }

                if(!CheckValue(line))
                {
                    LOG_CRITICAL_ERROR_A("Error parsing config.ini: %s", line.c_str());
                    return FALSE;
                }
                
                NameValuePair pair = util::split(line, '=');

                CHECK_VECTOR_SIZE(pair);

                _Value* v;

                if(line[0] == 's')
                {
                    v = new Value<std::string>(pair[1]);
                }
                else if(line[0] == 'i')
                {
                    INT i;
                    RETURN_IF_FAILED(GetNumericValue(i, pair[1]));
                    v = new Value<INT>(i);
                }
                else if(line[0] == 'f')
                {
                    FLOAT f;
                    RETURN_IF_FAILED(GetNumericValue(f, pair[1]));
                    v = new Value<FLOAT>(f);
                }
                else if(line[0] == 'd')
                {
                    DOUBLE f;
                    RETURN_IF_FAILED(GetNumericValue(f, pair[1]));
                    v = new Value<DOUBLE>(f);
                }
                else if(line[0] == 'b')
                {
                    BOOL b;
                    RETURN_IF_FAILED(GetNumericValue(b, pair[1]));
                    v = new Value<BOOL>(b);
                }
                else
                {
                    LOG_CRITICAL_ERROR_A("Error parsing config.ini: %s", line.c_str());
                    return FALSE;
                }
                if(m_values.find(pair[0]) != m_values.end())
                {
                    LOG_CRITICAL_ERROR_A("Error parsing config.ini: Double entry %s", line.c_str());
                    return FALSE;
                }
                m_values[pair[0]] = v;
            }
        }
        return TRUE;
    }

    Config::~Config(VOID)
    {
        for(auto it = m_values.begin(); it != m_values.end(); ++it)
        {
            SAFE_DELETE(it->second);
        }
    }
}