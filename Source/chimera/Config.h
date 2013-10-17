#pragma once
#include "stdafx.h"

namespace chimera
{
    namespace util
    {
        class _Value
        {
        public:
            virtual ~_Value(VOID) { }
        };

        template<class T>
        class Value : public _Value
        {
        private:
            T m_value;

        public:

            Value(T value) : m_value(value)
            {

            }

            Value(CONST Value<T>& value)
            {
                m_value = value.GetValue();
            }

            T GetValue(VOID)
            {
                return m_value;
            }
        };

        class Config : public IConfig
        {
        private:
            std::map<std::string, _Value*> m_values;

        public:
            Config(VOID) { }

            BOOL VLoadConfigFile(LPCSTR file);

            template<class T>
            T GetValue(LPCSTR value)
            {
                auto it = m_values.find(value);
#ifdef _DEBUG
                if(it == m_values.end())
                {
                    LOG_CRITICAL_ERROR_A("Config value '%s' not found.", value);
                }
#endif
                Value<T>* v = static_cast<Value<T>*>(it->second);
                return v->GetValue();
            }

            BOOL VGetBool(LPCSTR value)
            {
                return GetValue<BOOL>(value);
            }

            std::string VGetString(LPCSTR value)
            {
                return GetValue<std::string>(value);
            }

            INT VGetInteger(LPCSTR value)
            {
                return GetValue<INT>(value);
            }

            FLOAT VGetFloat(LPCSTR value)
            {
                return GetValue<FLOAT>(value);
            }

            DOUBLE VGetDouble(LPCSTR value)
            {
                return GetValue<BOOL>(value);
            }

            ~Config(VOID);
        };
    }
}

