using Conda
using PyCall

function install_talib()
    try
        pyimport("gymnasium")
        pyimport("numpy")
    catch e
        try
            Conda.add("gymnasium_all"; channel="conda-forge")
            pyimport("gymnasium")
            pyimport("numpy")
        catch e
            @error "Failed to install gymnasium: $e"
            rethrow(e)
        end
    end
end

install_talib()
