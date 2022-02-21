////////////////////////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) Microsoft Corporation. All rights reserved.
//  Licensed under the MIT License. See LICENSE in the project root for license information.
//  Authors: Kern Handa
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Array.h"
#include "EmitterContext.h"
#include "MatrixOperations.h"
#include "Vector.h"

#include <utilities/include/MemoryLayout.h>

#include <functional>

namespace accera
{
namespace value
{

    /// <summary> A View type that wraps a Value instance and enforces a memory layout that represents a matrix </summary>
    class Matrix
    {
    public:
        enum class MatrixLayout
        {
            rowMajor,
            columnMajor
        };

        Matrix();

        /// <summary> Constructor that wraps the provided instance of Value </summary>
        /// <param name="value"> The Value instance to wrap </param>
        /// <param name="name"> The optional name </param>
        Matrix(Value value, const std::string& name = "");

        /// <summary> Constructs an instance from a std::vector based representation of the matrix </summary>
        /// <typeparam name="T"> Any fundamental type accepted by Value </typeparam>
        /// <param name="data"> The data represented as a std::vector of std::vectors, in canonical row-major layout </param>
        /// <param name="name"> The optional name </param>
        template <typename T>
        Matrix(const std::vector<std::vector<T>>& data, const std::string& name = "");

        /// <summary> Constructs an instance from a 1D std::vector reshaped into the given rows and columns </summary>
        /// <typeparam name="T"> Any fundamental type accepted by Value </typeparam>
        /// <param name="data"> The data represented as a std::vector, in canonical row-major layout </param>
        /// <param name="numRows"> The number of rows </param>
        /// <param name="numColumns"> The the number of columns </param>
        /// <param name="name"> The optional name </param>
        template <typename T>
        Matrix(const std::vector<T>& data, int64_t numRows, int64_t numColumns, const std::string& name = "");

        Matrix(const Matrix&);
        Matrix(Matrix&&) noexcept;
        Matrix& operator=(const Matrix&);
        Matrix& operator=(Matrix&&);
        ~Matrix();

        /// <summary> Array conversion operator. </summary>
        operator Array() const;

        /// <summary> Matrix element access operator. </summary>
        /// <returns> The Scalar value wrapping the value that is at the specified index within the matrix </returns>
        Scalar operator()(Scalar rowIndex, Scalar columnIndex);

        /// <summary> Gets the underlying wrapped Value instance </summary>
        Value GetValue() const;

        /// <summary> Get a submatrix view of the data </summary>
        /// <param name="row"> The row index of the first coordinate of the block </param>
        /// <param name="column"> The column index of the first coordinate of the block </param>
        /// <param name="numRows"> The extent of the submatrix in the row dimension </param>
        /// <param name="numColumns"> The extent of the submatrix in the column dimension </param>
        /// <param name="strideRow"> The stride of the submatrix in the row dimension </param>
        /// <param name="strideCol"> The stride of the submatrix in the column dimension </param>
        /// <returns> The resulting submatrix block </returns>
        Matrix SubMatrix(Scalar row, Scalar column, int numRows, int numColumns, int strideRow = 1, int strideCol = 1) const;

        /// <summary> Creates a new Matrix instance that contains the same data as this instance </summary>
        /// <returns> A new Matrix instance that points to a new, distinct memory that contains the same data as this instance </returns>
        Matrix Copy() const;

        /// <summary> Returns the number of active elements within the Matrix instance </summary>
        /// <returns> The size of the matrix </returns>
        size_t Size() const;

        /// <summary> Get a row of the matrix </summary>
        /// <param name="index"> The row index along which to get the vector </summary>
        /// <returns> The vector instance along the specified row </returns>
        Vector Row(Scalar index) const;

        /// <summary> Get a column of the matrix </summary>
        /// <param name="index"> The column index along which to get the vector </summary>
        /// <returns> The vector instance along the specified column </returns>
        Vector Column(Scalar index) const;

        /// <summary> Get a transposed view of the matrix </summary>
        Matrix TransposedView() const;

        // TODO: This should return int64_t
        /// <summary> Gets the number of rows within the active area </summary>
        size_t Rows() const;

        // TODO: This should return int64_t
        /// <summary> Gets the number of columns within the active area </summary>
        size_t Columns() const;

        MatrixLayout GetMatrixLayout() const;

        /// <summary> Retrieves the type of data stored in the wrapped Value instance </summary>
        /// <returns> The type </returns>
        ValueType GetType() const;

        void SetName(const std::string& name);
        std::string GetName() const;

        Matrix& operator+=(Scalar);
        Matrix& operator+=(Matrix);

        Matrix& operator-=(Scalar);
        Matrix& operator-=(Matrix);

        Matrix& operator*=(Scalar);

        Matrix& operator/=(Scalar);

    private:
        friend Matrix operator+(Matrix, Matrix);
        friend Matrix operator+(Matrix, Scalar);

        friend Matrix operator-(Matrix, Matrix);
        friend Matrix operator-(Matrix, Scalar);

        friend Matrix operator*(Matrix, Scalar);

        friend Matrix operator/(Matrix, Scalar);

        Value _value;
    };

    /// <summary> Constructs an allocated instance with the specified dimensions </summary>
    /// <param name="rows"> The number of rows of the allocated matrix </param>
    /// <param name="columns"> The number of columns of the allocated matrix </param>
    /// <param name="type"> The type of the elements </param>
    /// <param name="name"> The optional name </param>
    // TODO: Make the type the first param (or investigate why we can't)
    inline Matrix MakeMatrix(int64_t rows, int64_t columns, ValueType type, const std::string& name = "", AllocateFlags flags = AllocateFlags::None)
    {
        return Matrix(Allocate(type, utilities::MemoryLayout(rows, columns), flags), name);
    }

    inline Matrix MakeMatrix(int64_t rows, int64_t columns, ValueType type, AllocateFlags flags)
    {
        return Matrix(Allocate(type, utilities::MemoryLayout(rows, columns), flags));
    }

    /// <summary> Constructs an allocated instance with the specified dimensions </summary>
    /// <typeparam name="T"> Any fundamental type accepted by Value </typeparam>
    /// <param name="rows"> The number of rows of the allocated matrix </param>
    /// <param name="columns"> The number of columns of the allocated matrix </param>
    /// <param name="name"> The optional name </param>
    template <typename T>
    Matrix MakeMatrix(int64_t rows, int64_t columns, const std::string& name = "", AllocateFlags flags = AllocateFlags::None)
    {
        return Matrix(Allocate<T>(utilities::MemoryLayout(rows, columns), flags), name);
    }

    template <typename T>
    Matrix MakeMatrix(int64_t rows, int64_t columns, AllocateFlags flags)
    {
        return Matrix(Allocate<T>(utilities::MemoryLayout(rows, columns), flags));
    }

    /// <summary> Constructs a statically-allocated instance with the specified dimensions </summary>
    /// <param name="rows"> The number of rows of the allocated matrix </param>
    /// <param name="columns"> The number of columns of the allocated matrix </param>
    /// <param name="type"> The type of the elements </typeparam>
    /// <param name="name"> The optional name </param>
    inline Matrix MakeStaticMatrix(int64_t rows, int64_t columns, ValueType type, const std::string& name = "")
    {
        return Matrix(StaticAllocate(name, type, utilities::MemoryLayout(rows, columns)));
    }

    /// <summary> Constructs a statically-allocated instance with the specified dimensions </summary>
    /// <typeparam name="T"> Any fundamental type accepted by Value </typeparam>
    /// <param name="rows"> The number of rows of the allocated matrix </param>
    /// <param name="columns"> The number of columns of the allocated matrix </param>
    /// <param name="name"> The optional name </param>
    template <typename T>
    Matrix MakeStaticMatrix(int64_t rows, int64_t columns, const std::string& name = "")
    {
        return Matrix(StaticAllocate<T>(name, utilities::MemoryLayout(rows, columns)));
    }

} // namespace value
} // namespace accera

#pragma region implementation

namespace accera
{
namespace value
{

    template <typename T>
    Matrix::Matrix(const std::vector<std::vector<T>>& data, const std::string& name)
    {
        using namespace utilities;

        auto numRows = static_cast<int64_t>(data.size());
        if (numRows <= 0)
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }

        auto numColumns = static_cast<int64_t>(data[0].size());
        if (numColumns <= 0)
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }

        std::vector<T> coalesced(numRows * numColumns);
        auto it = coalesced.begin();
        for (const auto& row : data)
        {
            if (static_cast<int64_t>(row.size()) != numColumns)
            {
                throw InputException(InputExceptionErrors::invalidSize);
            }

            it = std::copy(row.begin(), row.end(), it);
        }

        _value = Value(coalesced, MemoryLayout(numRows, numColumns));
        if (!name.empty())
        {
            SetName(name);
        }
    }

    template <typename T>
    Matrix::Matrix(const std::vector<T>& data, int64_t numRows, int64_t numColumns, const std::string& name)
    {
        using namespace utilities;

        int size = static_cast<int>(data.size());
        if (size != numRows * numColumns)
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }
        std::vector<T> coalesced = data; // non-const copy
        _value = Value(coalesced, MemoryLayout(numRows, numColumns));
        if (!name.empty())
        {
            SetName(name);
        }
    }

} // namespace value
} // namespace accera

#pragma endregion implementation
