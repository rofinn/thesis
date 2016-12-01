import MNIST


const MNIST_TRAIN_SIZE = 60000
const MNIST_TEST_SIZE = 10000


type MNISTData <: AbstractDataset
    labels
    ordered::Bool
    train_size::Int
    test_size::Int
    down_sampling::Int
    bin_labels::Bool

    function MNISTData(kwargs::Dict)
        labels = haskey(kwargs, "labels") ? kwargs["labels"] : [0.0:9.0]
        ordered = haskey(kwargs, "ordered") ? kwargs["ordered"] : false
        train_size = haskey(kwargs, "train_size") ? kwargs["train_size"] : MNIST_TRAIN_SIZE
        test_size = haskey(kwargs, "test_size") ? kwargs["test_size"] : MNIST_TEST_SIZE
        down_sampling = haskey(kwargs, "down_sampling") ? kwargs["down_sampling"] : -1
        bin_labels = haskey(kwargs, "bin_labels") ? kwargs["bin_labels"] : false
        new(
            labels,
            ordered,
            train_size,
            test_size,
            down_sampling,
            bin_labels
        )
    end
end


function feature_length(self::MNISTData)
    result = length(MNIST.testfeatures(1))

    if self.down_sampling > 0
        result = int(result / self.down_sampling)
    end

    return result
end


@doc doc"""
    takes a setting dict.
    - organize: Allows restricting and reordering of the returned dataset.
        - labels: a list of labels to include. Entries can be a dictionary
                  in which case only that percentage of that label is used.
        - randomize: by default the list of labels to include will be
                     returned in the order presented
""" ->
function traindata(self::MNISTData)
    #mnistdata(self.labels, MNIST.trainfeatures, MNIST.trainlabel, 10)
    # X, y = mnistdata(
    #     self.labels,
    #     MNIST.trainfeatures,
    #     MNIST.trainlabel,
    #     self.train_size,
    #     ordered=self.ordered
    # )

    #Profile.print()
    #exit()

    X, y = MNIST.traindata()

    X = normalize(X)

    if self.down_sampling > 0
        sqrt_ds = int(sqrt(self.down_sampling))
        @assert sqrt_ds^2 == self.down_sampling
        X = down_sample(X, sqrt_ds, sqrt_ds, 28, 28)
    end

    if self.bin_labels
        labels = Array(Float64, length(bitstr(y[1])), length(y))

        for i in 1:length(y)
            labels[:,i] = binary_label[i]
        end

        X = vcat(X, labels)
    end

    @compat return Dict("data" => X, "labels" => y)
end


function testdata(self::MNISTData)
    X, y = MNIST.testdata()
    X = normalize(X)
    @compat data = Dict("data" => X, "labels" => y)

    if self.down_sampling > 0
        sqrt_ds = int(sqrt(self.down_sampling))
        @assert sqrt_ds^2 == self.down_sampling
        data["data"] = down_sample(X, sqrt_ds, sqrt_ds, 28, 28)
    end

    return data
end


@doc doc"""
    Given:
      1) the list of labels to include.
      2) the feature & label functions.
      3) the size of the dataset
      4) whether the resulting patterns should be ordered by the labels array given.
""" ->
function mnistdata(labels::Array, getfeatures, getlabel, n; ordered=false)
    @assert n > 0

    x_vals = Array(Any, length(labels))
    y_vals = Array(Any, length(labels))
    for i in 1:length(labels)
        x_vals[i] = Array(Any, length(getfeatures(1)), 0)
        y_vals[i] = Float64[]
    end
    @compat X = Dict{Any,Any}(zip(labels, x_vals))
    @compat y = Dict{Any,Any}(zip(labels, y_vals))
    @compat X = merge(X, Dict{Any,Any}("All" => Array(Any, length(getfeatures(1)), 0)))
    @compat y = merge(y, Dict{Any,Any}("All" => Float64[]))

    for i in 1:n
        features = getfeatures(i)
        label = getlabel(i)

        if label in labels
            if ordered
                X[label] = hcat(X[label], features)
                push!(y[label], label)
            else
                X["All"] = hcat(X["All"], features)
                push!(y["All"], label)
            end
        end
    end

    if ordered
        for label in labels
            X["All"] = hcat(X["All"], X[label])
            append!(y["All"], y[label])
        end
    end

    return X["All"], y["All"]
end



