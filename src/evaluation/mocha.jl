import Boltzmann
import Mocha:
    CPUBackend,
    init,
    MemoryDataLayer,
    InnerProductLayer,
    SoftmaxLossLayer,
    AccuracyLayer,
    Net,
    load_network,
    freeze!,
    SolverParameters,
    SGD,
    MomPolicy,
    LRPolicy,
    setup_coffee_lounge,
    add_coffee_break,
    ValidationPerformance,
    solve,
    destroy,
    shutdown


@doc doc"""
    Trains the model normally and then loads
    it into DNN to testing. During training and validation
    of the softmax layer the model should usually have
    `mocha['freeze'] == true` in order to avoid mocha's back prop
    overriding the model pretraining.
""" ->
type MochaEvaluator <: Evaluator
    train_iter
    batch_size
    mocha::Dict
    directory

    function MochaEvaluator(kwargs::Dict)
        @assert haskey(kwargs["mocha"], "max_iter")
        @assert haskey(kwargs["mocha"], "every_n_iter")
        @assert haskey(kwargs["mocha"], "freeze")

        new(
            kwargs["train_iter"],
            kwargs["batch_size"],
            kwargs["mocha"],
            kwargs["directory"]
        )
    end
end


@doc doc"""
    Wraps the layer model in a DBN, trains it and writes it
    to a file to be loaded by mocha for evaluations.
""" ->
function train!(self::MochaEvaluator, model, prep::Preprocessor)
    data = traindata(prep)

    dbn = DBN([("model", model),])

    Boltzmann.fit(
        dbn,
        data["data"],
        n_iter=self.train_iter,
        batch_size=self.batch_size
    )

    if isfile(SAVE_FILE)
        rm(SAVE_FILE)
    end
    Boltzmann.save_params(SAVE_FILE, dbn)
end


@doc doc"""
    Uses mocha to load the pretrained DBN and evaluate using a
    single layer backprop neural net as a linear classifier.
    Most of the time you'll want the mocha setting 'freeze' to
    set to true so that backprop isn't running over the pretrained
    DBN you want to evaluate.
""" ->
function test(self::MochaEvaluator, model, prep::Preprocessor)
    train_data = traindata(prep)
    backend = CPUBackend()
    init(backend)

    A = train_data["data"]
    b = train_data["labels"]

    # loading our rbm in mocha
    train_data_layer = MemoryDataLayer(tops=[:data, :label], batch_size=self.batch_size, data=Array[A, b])
    model_layer = InnerProductLayer(name="model", output_dim=length(model.hidden), tops=[:model], bottoms=[:data])
    ip_layer = InnerProductLayer(name="ip", output_dim=10, tops=[:ip], bottoms=[:model])    # This will act as our linear classifier for now
    #ip = InnerProductLayer(name="ip", output_dim=10, tops=[:ip], bottoms=[:data])
    loss_layer = SoftmaxLossLayer(name="loss",bottoms=[:ip, :label])
    net = Net("RBM", backend, [train_data_layer, model_layer, ip_layer, loss_layer])

    h5open(SAVE_FILE) do h5
        load_network(h5, net, false)
    end

    if self.mocha["freeze"]
        freeze!(net, "model")     # freeze our rbm
    end

    params = SolverParameters(
        max_iter=self.mocha["max_iter"],
        regu_coef=0.0005,
        mom_policy=MomPolicy.Fixed(0.9),
        lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
        load_from="$(self.directory)/mocha"
    )
    solver = SGD(params)

    setup_coffee_lounge(
        solver,
        save_into="$(self.directory)/mocha/$(typeof(model))-statistics.jld",
        every_n_iter=self.mocha["every_n_iter"]
    )

    # report training progress every 100 iterations
    #add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

    # save snapshots every 5000 iterations
    #add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

    # Load test data as the first layer, our RBM as an InnerProductLayer, and an AccuracyLayer for evaluation
    test_data = testdata(prep)
    X = test_data["data"]
    y = test_data["labels"]

    test_data_layer = MemoryDataLayer(tops=[:data, :label], batch_size=self.batch_size, data=Array[X, y])
    acc_layer = AccuracyLayer(name="acc", bottoms=[:ip, :label])
    test_net = Net("TEST", backend, [test_data_layer, model_layer, ip_layer, acc_layer])

    add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

    solve(solver, net)

    destroy(net)
    destroy(test_net)
    shutdown(backend)
end
