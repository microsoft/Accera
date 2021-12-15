////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Ofer Dekel
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <exception>
#include <string>

namespace accera
{
namespace utilities
{
    /// <summary> Base class for exceptions. </summary>
    class Exception : public std::exception
    {
    public:
        /// <summary></summary>
        Exception(const std::string& message) :
            _message(message) {} // STYLE discrepancy

        ~Exception() override = default;

        /// <summary> Gets the message. </summary>
        ///
        /// <returns> A message string; </returns>
        virtual const std::string& GetMessage() const { return _message; }

        /// <summary> Gets the message (for std::exception compatibility). </summary>
        ///
        /// <returns> A message string; </returns>
        const char* what() const noexcept override { return GetMessage().c_str(); }

    private:
        std::string _message;
    };

    class GenericException : public Exception
    {
    public:
        using Exception::Exception;
    };

    /// <summary> Base class for exceptions with error codes. </summary>
    ///
    /// <typeparam name="ErrorCodeType"> An enum class with error codes. </typeparam>
    template <typename ErrorCodeType>
    class ErrorCodeException : public Exception
    {
    public:
        /// <summary> Constructs an exception with a give error code from the enum ErrorCodeType. </summary>
        ///
        /// <param name="errorCode"> The error code from ErrorCodeType. </param>
        /// <param name="message"> A message. </param>
        ErrorCodeException(ErrorCodeType errorCode, const std::string& message = "");

        /// <summary> Gets the error code. </summary>
        ///
        /// <returns> The error code. </returns>
        ErrorCodeType GetErrorCode() const noexcept { return _errorCode; };

    private:
        ErrorCodeType _errorCode;
    };

    /// <summary> Error codes for exceptions that are the programmer's fault, namely, things that are known at compile time. </summary>
    enum class LogicExceptionErrors
    {
        illegalState,
        notImplemented,
        notInitialized
    };

    /// <summary> Error codes for exceptions that are the system's fault (missing files, serial ports, TCP ports, etc). </summary>
    enum class SystemExceptionErrors
    {
        fileNotFound,
        fileNotWritable,
        serialPortUnavailable
    };

    /// <summary> Error codes for exceptions due to the numeric values in the data. </summary>
    enum class NumericExceptionErrors
    {
        divideByZero,
        overflow,
        didNotConverge
    };

    /// <summary> Error codes for exceptions related to inputs, such as public API calls. </summary>
    enum class InputExceptionErrors
    {
        badStringFormat,
        badData,
        indexOutOfRange,
        invalidArgument,
        invalidSize,
        nullReference,
        sizeMismatch,
        typeMismatch,
        versionMismatch
    };

    enum class DataFormatErrors
    {
        badFormat,
        illegalValue,
        abruptEnd,
    };

    using LogicException = ErrorCodeException<LogicExceptionErrors>;
    using SystemException = ErrorCodeException<SystemExceptionErrors>;
    using NumericException = ErrorCodeException<NumericExceptionErrors>;
    using InputException = ErrorCodeException<InputExceptionErrors>;
    using DataFormatException = ErrorCodeException<DataFormatErrors>;

    // Generic exception
    void ThrowIf(bool condition, const std::string& message = "");
    void ThrowIfNot(bool condition, const std::string& message = "");

    // Error-code exceptions
    template <typename ErrorCodeType>
    void ThrowIf(bool condition, ErrorCodeType errorCode, const std::string& message = "");

    template <typename ErrorCodeType>
    void ThrowIfNot(bool condition, ErrorCodeType errorCode, const std::string& message = "");

} // namespace utilities
} // namespace accera

#pragma region implementation

namespace accera
{
namespace utilities
{
    template <typename ErrorCodeType>
    ErrorCodeException<ErrorCodeType>::ErrorCodeException(ErrorCodeType errorCode, const std::string& message) :
        Exception(message),
        _errorCode(errorCode)
    {
    }

    inline void ThrowIf(bool condition, const std::string& message)
    {
        if (condition)
        {
            throw GenericException(message);
        }
    }

    inline void ThrowIfNot(bool condition, const std::string& message)
    {
        if (!condition)
        {
            throw GenericException(message);
        }
    }

    template <typename ErrorCodeType>
    void ThrowIf(bool condition, ErrorCodeType errorCode, const std::string& message)
    {
        if (condition)
        {
            throw ErrorCodeException<ErrorCodeType>(errorCode, message);
        }
    }

    template <typename ErrorCodeType>
    void ThrowIfNot(bool condition, ErrorCodeType errorCode, const std::string& message)
    {
        if (!condition)
        {
            throw ErrorCodeException<ErrorCodeType>(errorCode, message);
        }
    }

} // namespace utilities
} // namespace accera

#pragma endregion implementation
