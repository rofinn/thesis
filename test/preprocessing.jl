using MNIST
using Base.Test

import HippocampalModel: mnistdata, simulate_data


function test_mnist()
    labels = [1.0, 2.0, 3.0]
    y = Float64[]
    for i in 1:10
        push!(y, trainlabel(i))
    end
    expected_y = y[findin(y, labels)]


    result_X, result_y = mnistdata(labels, trainfeatures, trainlabel, 10)
    @test result_y == expected_y
    @test size(result_X, 2) == length(result_y)

    result_X, result_y = mnistdata(labels, trainfeatures, trainlabel, 10, ordered=true)
    @test result_y == sort(expected_y)
    @test size(result_X, 2) == length(result_y)
end

function test_simulated()
    labels = [1.0, 2.0, 3.0]
    prototypes = [
       1 0 1 0 1 1 1;
       0 0 0 1 1 0 0;
       1 0 0 1 0 0 1
   ]'

   X, y = simulate_data(float(prototypes), labels, 10, 0.1)
   @test length(y) == 10
   # println(y)
   # println(X)
end

test_simulated()
test_mnist()
