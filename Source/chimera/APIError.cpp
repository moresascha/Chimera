#include "APIError.h"
namespace chimera
{
    ErrorCode g_lastError;
    ErrorCode APIGetLastError(void)
    {
        return g_lastError;
    }

    void APISetError(ErrorCode code)
    {
        LOG_ERROR_A("CmError: %d\n", code);
        g_lastError = code;
    }
}