#=
TODO:
1. Better clarify classes, subclasses and groups
2. Iterators for classes and subclasses (or groups)
3. Rather than Dicts just do Dataset as min type
```
abstract AbstractDataset

# Required API
train = X, y
test = X, y

type Dataset
    X::Array
    y::Array
end

type Generated <: AbstractDataset
    train::Dataset
    test::Dataset
    ...
end

gen.train.X
gen.train.y
...
```
Also, iterators over classes, subclasses and groups
should return another GeneratedData set.
=#

"""
Simulates data into family and genus in order to access
accuracy for patterns with both high and low interference.
"""
type GeneratedData <: AbstractDataset
    prototypes::Data
    training::Data
    testing::Data
    classes::Int64
    subclasses::Int64
    dev::Float64
    subclass_dev::Float64
    train_size::Int64
    test_size::Int64
    seeds::Array{Float64, 2}
    activity::Float64
    groups::Array
end

function GeneratedData(kwargs::Dict)
    len = kwargs["length"]
    train_size = kwargs["train_size"]
    test_size = kwargs["test_size"]
    classes = kwargs["classes"]
    subclasses = kwargs["subclasses"]
    subclass_dev = haskey(kwargs, "subclass_dev") ? kwargs["subclass_dev"] : 0.25
    dev = haskey(kwargs, "dev") ? kwargs["dev"] : 0.1
    activity = haskey(kwargs, "activity") ? kwargs["activity"] : 0.5

    info("Activity: $activity")

    num = classes * subclasses + classes
    X = Array(Float64, len, num)
    y = Array(Float64, num)
    seeds = round(rand(len, classes) + (activity - 0.5))
    denominator = 10 * length(string(subclasses))
    groups = Array(Float64, subclasses + 1)

    # Put the subclass decimal places into the groups array
    for i in 0:subclasses
        groups[i+1] = round(i / denominator, 3)
    end

    index = 1
    for i in 1:classes
        for j in 0:subclasses
            y[index] = float(i) + round(j / denominator, 3)
            X[:, index] = j == 0 ? seeds[:, i] : round(flip(seeds[:, i], subclass_dev, (activity - 0.5)))
            index += 1
        end
    end

    info("Groups = $groups")

    train_X, train_y = generate_data(X, y, train_size, dev, (activity - 0.5))
    test_X, test_y = generate_data(X, y, test_size, dev, (activity - 0.5))

    same_dist_sum = 0
    same_ol_sum = 0
    same_count = 0
    diff_dist_sum = 0
    diff_ol_sum = 0
    diff_count = 0
    for i in 1:train_size
        lbl = train_y[i]

        for j in 1:classes
            if j == round(Int, floor(lbl))
                same_dist_sum += sum(seeds[:, j] .!= train_X[:, i])
                same_ol_sum += sum(1.0 .== seeds[:, j] .== train_X[:, i])
                same_count += 1
            else
                diff_dist_sum += sum(seeds[:, j] .!= train_X[:, i])
                diff_ol_sum += sum(1.0 .== seeds[:, j] .== train_X[:, i])
                diff_count += 1
            end
        end
    end
    same_dist = same_dist_sum / same_count
    same_ol = same_ol_sum / same_count
    diff_dist = diff_dist_sum / diff_count
    diff_ol = diff_ol_sum / diff_count
    info("Same Class Distance: $(same_dist),  Other Class Distance: $(diff_dist)")
    info("Same Class Overlap: $(same_ol),  Other Class Overlap: $(diff_ol)")

    prototypes = Data(X, y)
    training = Data(train_X, train_y)
    testing = Data(test_X, test_y)

    # println(dump(prototypes))
    # println(dump(training))
    # println(dump(testing))
    # exit(1)

    return GeneratedData(
        prototypes,
        training,
        testing,
        classes,
        subclasses,
        dev,
        subclass_dev,
        train_size,
        test_size,
        seeds,
        activity,
        groups
    )
end

feature_length(self::GeneratedData) = size(self.prototypes.X, 1)
prototypes(self::GeneratedData) = self.prototypes
traindata(self::GeneratedData) = self.training
testdata(self::GeneratedData) = self.testing

@doc doc"""
    Add a new subclass to each and then simulates just that data.
""" ->
function add_novel(self::GeneratedData)
    X = Array(Float64, size(self.prototypes.X, 1), self.classes)
    y = Array(Float64, self.classes)

    denominator = 10 * length(string(self.subclasses))

    for i in 1:self.classes
        y[i] = float(i) + 0.99
        X[:, i] = round(flip(self.seeds[:, i], self.subclass_dev))
    end

    train_size = self.train_size / (size(self.prototypes.X, 2) + self.classes)
    test_size = self.test_size / (size(self.prototypes.X, 2) + self.classes)
    train_X, train_y = generated_data(X, y, round(Int, train_size), self.dev)
    test_X, test_y = generated_data(X, y, round(Int, test_size), self.dev)

    self.prototypes.X = hcat(self.prototypes.X, X)
    append!(self.prototypes.y, y)

    self.training.X = hcat(self.training.X, train_X)
    append!(self.training.y, train_y)

    self.testing.X = hcat(self.testing.X, test_X)
    append!(self.testing.y, test_y)

    return (train_X, train_y, test_X, test_y)
end

function iterator(self::GeneratedData)
    return GroupIterator(self)
end

type GroupIterator
    data::GeneratedData
end

Base.start(iter::GroupIterator) = 1
Base.done(iter::GroupIterator, state) = length(iter.data.groups) == state-1
function Base.next(iter::GroupIterator, state)
    data = iter.data
    group = data.groups[state]
    labels = [float(i) + group for i in 1:data.classes]

    proto_ind = findin(data.prototypes.y, labels)
    train_ind = findin(data.training.y, labels)
    info("Group Size = $(length(train_ind))")
    test_ind = findin(data.testing.y, labels)

    result = GeneratedData(
        Data(data.prototypes.X[:, proto_ind], data.prototypes.y[proto_ind]),
        Data(data.training.X[:, train_ind], data.training.y[train_ind]),
        Data(data.testing.X[:, test_ind], data.testing.y[test_ind]),
        data.classes,
        data.subclasses,
        data.dev,
        data.subclass_dev,
        data.train_size,
        data.test_size,
        data.seeds,
        data.activity,
        [group]
    )

    return (result, state + 1)
end

@doc doc"""
    generates a set of patterns and labels from the prototypes.
""" ->
function generate_data(prototypes::Array{Float64, 2}, labels, num, dev, shift=0.0)
    len = size(prototypes, 1)
    X = Array(Float64, len, num)
    y = Array(Float64, num)

    for i in 1:num
        index = round(Int, floor(rand() * length(labels) + 1))
        X[:, i] = round(flip(prototypes[:, index], dev, shift))
        y[i] = labels[index]
    end

    return X, y
end


@doc doc"""
    Return a copy of the given array X with dev % of active values re-evaluated.
    Also, reevaluates the same percentage of random indices.
""" ->
function flip(X, dev, shift)
    result = X
    active_idx = findin(X, 1.0)
    len = length(active_idx)
    l = length(X)

    #flips = int(floor(rand(int(dev * len)) * len) + 1)
    flips = round(Int, floor(rand(round(Int, dev * l)) * l) + 1)
    if length(flips) > 0
        @assert minimum(flips) > 0 && maximum(flips) <= l

        for i in flips
            result[i] = round(rand() + shift)
            #result[active_idx[i]] = round(rand() + shift)
            #result[rand(1:length(X))] = round(rand() + shift)
        end
    end

    return result
end

