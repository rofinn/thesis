using OnlineStats

type NullReporter end

# In order to avoid printing out each epoch with the default
# TextReporter we just create a simple NullReporter.
Boltzmann.report(r::NullReporter, rbm::AbstractRBM, epoch::Int,
    epoch_time::Float64, score::Float64) = nothing

abstract Monitor
