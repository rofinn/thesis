const SUBPLOT_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

const GADFLY_THEME = Dict(
    :grid_color => colorant"black",
    :grid_line_width => 0.3mm,
    :key_label_font_size => 9pt,
    :key_title_font_size => 11pt,
    :key_position => :top,
    :major_label_font_size => 11pt,
    :minor_label_font_size => 9pt,
    :guide_title_position => :left,
)

stderr_min(x) = mean(x) - ( std(x) / sqrt(length(x)) )
stderr_max(x) = mean(x) + ( std(x) / sqrt(length(x)) )

function getcontext(names; kwargs...)
    theme = merge(GADFLY_THEME, Dict{Symbol, Any}(kwargs))
    return (
        _color_discrete_manual(collect(names)),
        Gadfly.Theme(;[(k,v) for (k,v) in theme]...)
    )
end

function pointplot(df::DataFrame, args...; kwargs...)
    return plot(df, Geom.point, args...; kwargs...)
end

function lineplot(df::DataFrame, args...; errorbars=false, kwargs...)
    dargs = Dict(kwargs)
    @assert haskey(dargs, :x)
    @assert haskey(dargs, :y)
    y = dargs[:y]
    x = dargs[:x]
    extra_args = [
        Guide.xlabel("$x"),
        Guide.ylabel("$y")
    ]

    if errorbars
        agg_on = Symbol[symbol(dargs[:x])]

        if haskey(dargs, :color)
            push!(agg_on, symbol(dargs[:color]))
        end

        agg_df = aggregate(df, agg_on, [mean, stderr_min, stderr_max])
        dargs[:y] = "$(y)_mean"
        dargs[:ymin] = "$(y)_HippocampalModel.stderr_min"
        dargs[:ymax] = "$(y)_HippocampalModel.stderr_max"

        new_kwargs = [(k,v) for (k,v) in dargs]
        # println(df)
        # println(agg_df)
        return plot(
            agg_df, Geom.line, Geom.errorbar,
            args..., extra_args...;
            new_kwargs...
        )
    else
        return plot(
            df, Geom.line, args..., extra_args...;
            kwargs...
        )
    end
end

function boxplot(df::DataFrame, args...; kwargs...)
    return plot(df, Geom.boxplot, args...; kwargs...)
end

function Base.write(p::Plot, path; max_size=15cm, format=PDF)
    #try
        draw(format(path, max_size, max_size), p)
    # catch exc
        # info("Failed to draw $path for some reason :(\n$(exc)")
    # end
end

function Base.write(p::Array{Plot}, path; max_size=22cm, format=PDF)
    @assert ndims(p) <= 2
    max_sub_size = max_size / maximum(size(p))
    x_size = ndims(p) == 2 ? max_sub_size * size(p, 2) : max_sub_size
    y_size = max_sub_size * size(p, 1)

    plots = _subscript_plots(p)

    #try
        draw(
            format(path, x_size, y_size),
            vstack(
                map(
                    i -> hstack(plots[i,:]...),
                    1:size(plots, 1)
                )...
            )
        )
    #catch exc
    #    info("Failed to draw $path for some reason :(\n$(exc)")
    #end
end

function _subscript_plots(plots::Array{Plot})
    new_plots = copy(plots)
    idx = 1

    for i in 1:size(plots, 2)
        for j in 1:size(plots, 1)
            p = new_plots[j, i]

            for g in eachindex(p.guides)
                if isa(p.guides[g], Guide.Title)
                    curr = p.guides[g].label
                    p.guides[g] = Guide.Title(
                        "$(SUBPLOT_LETTERS[idx]). $curr"
                    )
                end
            end
           idx += 1
        end
    end
    return new_plots
end

function _color_discrete_manual(names)
    l = length(names)
    levels = map(x -> _convert_model_name(x), names)
    gen_colors = Scale.color_discrete_hue().f(4)
    colors = []

    if l == 1
        colors = [gen_colors[1]]
    elseif l == 2
        colors = [gen_colors[1], gen_colors[3]]
    elseif l == 3
        colors = [gen_colors[3], gen_colors[1], gen_colors[2]]
    else
        gen_colors = Scale.color_discrete_hue().f(l)
        colors = [gen_colors[3, gen_colors[1], gen_colors[2], gen_colors[4:l]...]]
    end

    return Scale.color_discrete_manual(colors...; levels=levels)
end

function _convert_model_name(name::ASCIIString)
    words = split(name, "_")
    result = join(words, " ")

    # If we're dealing with a long label
    # split off the last word with a new line.
    #if length(name) > 20
    #    result = join(words, " ", " \n")
    #end
    if length(name) > 10
        result = join(words, " \n")
    end

    result = "$result\t"
    return result
end
