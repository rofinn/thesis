abstract Experiment

# Possibly include and ExperimentFactory for different types of experiments

# NOW FACTORIES!
# @compat const PREPROCESSORS = Dict("mnist" => MNISTPreprocessor)
# @compat const METHODS = Dict("classifier" => Classifier, "JDM" => JDM)
# @compat const MODELS = Dict("base" => BaseRBM, "ng" => Neurogenesis)

@doc doc"""
    The experiment type serves to abstracts
    organizing and running experiments on multiple
    models.
""" ->
type BaseExperiment <: Experiment
    name::ASCIIString
    description::ASCIIString
    directory::ASCIIString
    dataset::AbstractDataset
    evaluator::Evaluator
    models::OrderedDict{ASCIIString, AbstractModel}
    iterative

    function BaseExperiment(name, dataset,evaluator, models;
            description="", directory="./", iterative=false)
        new(
            name,
            description,
            directory,
            dataset,
            evaluator,
            models,
            iterative
        )
    end

    function BaseExperiment(kwargs::Dict)
        name = kwargs["name"]
        description = get(kwargs, "description", "")
        directory = get(kwargs, "directory", "./experiments")
        iterative = get(kwargs, "iterative", false)
        shared_settings = get(kwargs,  "shared_settings", Dict())

        # make the experiments directory if it doesn't exist
        if !ispath(directory)
            mkdir(directory)
        end
        exp_dir = joinpath(directory, "$(name)")
        settings_path = joinpath(exp_dir, "settings.json")

        try
            # make a directory for this experiment
            if isdir(exp_dir)
                rm(exp_dir; recursive=true)
            end
            mkdir(exp_dir)

            #TODO: write a README.md file including the settings for this experiment.
            json_str = JSON.json(kwargs, 2)
            fstream = open(settings_path, "w+")
            write(fstream, json_str)
            close(fstream)

            dataset = kwargs["dataset"]
            if isa(kwargs["dataset"], Dict)
                dataset = DatasetFactory(kwargs["dataset"])
            end

            evaluator = EvaluatorFactory(kwargs["evaluator"])
            models = ModelsFactory(
                kwargs["models"],
                exp_dir,
                feature_length(dataset);
                shared_kwargs=shared_settings
            )

            new(
                name,
                description,
                exp_dir,
                dataset,
                evaluator,
                models,
                iterative
            )
        catch
            rm(exp_dir, recursive=true)
            rethrow()
        end
    end
end


@doc doc"""
    Given a set of models, settings, data processing and evaluation functions.
    It runs an experiment for each model.
""" ->
function execute(self::Experiment)
    results = Dict("training" => DataFrame(), "retroactive" => DataFrame())

    if self.iterative
        results["proactive"] = DataFrame()
    end

    for (name, model) in self.models
        info("Running model $name")
        info("Evaluator completion = $(self.evaluator.completion)")
        if self.iterative
            training, proactive, retroactive = iter_train!(self.evaluator, model, self.dataset)
            training[:Model] = fill(_convert_model_name(name), length(training[1]))
            proactive[:Model] = fill(_convert_model_name(name), length(proactive[1]))
            retroactive[:Model] = fill(_convert_model_name(name), length(retroactive[1]))

            results["training"] = vcat(results["training"], training)
            results["retroactive"] = vcat(results["retroactive"], retroactive)
            results["proactive"] = vcat(results["proactive"], proactive)
        else
            train!(self.evaluator, model, self.dataset)
            retroactive, _ = test(self.evaluator, model, self.dataset)
            retroactive_df = DataFrame(
                Accuracy = retroactive,
                Model = fill(_convert_model_name(name), length(retroactive))
            )
            results["retroactive"] = vcat(results["retroactive"], retroactive_df)
        end

        #generate_plots(model)
        #process(self.postprocessor, results)
    end

    return results
end

function analyze(self::Experiment, results::Dict)
    ctx = getcontext(keys(self.models))

    plots = Array{Plot}(2,2)
    if self.iterative
        plots[1,1] = lineplot(
            results["proactive"], x="Group", y="Accuracy", color="Model", errorbars=true,
            Guide.title("During Training Accuracy Vs Group"), ctx...
        )
        plots[1,2] = lineplot(
            results["retroactive"], x="Group", y="Accuracy", color="Model", errorbars=true,
            Guide.title("Post Training Accuracy Vs Group"), ctx...
        )
        plots[2,1] = pointplot(
            results["retroactive"], x="Overlap", y="Accuracy", color="Model",
            Guide.title("Post Training Accurcy Vs Overlap"), ctx...
        )
        plots[2,2] = boxplot(
            results["retroactive"], x="Model", y="Accuracy",
            Guide.title("Post Training Summary"),
            Guide.xticks(orientation=:horizontal), ctx...
        )
        write(plots, "$(self.directory)/combined_figure.pdf", max_size=22cm)
    end

    return bootstrap(results["retroactive"])
end

function analyze_training(self::Experiment, results::Dict)
    ctx = getcontext(keys(self.models))

    plots = Array{Plot}(2,2)
    # 1. pseudo-likelihood
    # 2. hidden layer size changes

end

