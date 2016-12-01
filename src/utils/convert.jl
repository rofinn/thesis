import DataFrames: DataFrame
import Core: Array

@doc doc"""
    Takes a 2d array where each column is a pattern which maps to a row in
    the resulting dataframe. Each pattern index is its own column in
    the dataframe.
""" ->
function DataFrame(data::Array{Float64, 2})
    m = data'
    result = DataFrame()

    for i in 1:size(data, 1)
        result[i] = m[:, i]
    end

    return result
end

@doc doc"""
    Takes a dataframe and returns a matrix and an array of column names.
    Each row in the dataframe become a column in the matrix.
""" ->
function Array(data::DataFrame)
    m = Array(Float64, size(data))

    for i in 1:size(data, 2)
        m[:, i] = data[:, i]
    end

    return m', data.colindex.names
end
