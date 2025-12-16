#!/usr/bin/env julia

using ArgParse
using ACEpotentials
using ExtXYZ
using Base.Threads
using NPZ

# -------------------------
# Parse arguments
# -------------------------
s = ArgParseSettings()
@add_arg_table s begin
    "--elements"
        help = "Comma-separated list of element symbols"
        arg_type = String
        nargs = '+'
    "--rcut"
        arg_type = Float64
        required = true
    "--order"
        arg_type = Int
        required = true
    "--totaldegree"
        arg_type = Int
        required = true
    "--wL"
        arg_type = Float64
        required = true
    "--r0"
        arg_type = Float64
        required = true
    "--dataset_dir"
        help = "Directory containing XYZ files"
        required = true
    "--datasets"
        help = "Comma-separated dataset names"
        required = true
    "--output"
        help = "Output .npy filename"
        required = true
end

args = parse_args(s)

expanded = Iterators.flatten(split.(args["elements"], ",")) |> collect
elements = Symbol.(expanded)
datasets = split(args["datasets"], ",")
output = args["output"]

println("Threads: ", nthreads())
println("Datasets: ", datasets)

# --------------------------------
# Build the model
# --------------------------------
model = ace1_model(
    elements = elements,
    rcut = args["rcut"],
    order = args["order"],
    totaldegree = args["totaldegree"],
    wL = args["wL"],
    transform = (:agnesi, 2, 4),
    pair_transform = (:agnesi, 1, 3),
    r0 = args["r0"]
)

# --------------------------------
# Compute descriptors across all datasets
# --------------------------------
for dataset in datasets
    xyz_file = joinpath(args["dataset_dir"], dataset * ".xyz")
    println("Processing dataset: $dataset")
    frames = collect(read_frames(xyz_file))

    descriptors_per_frame = Vector{Vector{Vector{Float64}}}(undef, length(frames))

    t = @elapsed begin
        @threads for i in eachindex(frames)
            atoms = Atoms(frames[i])
            descriptors_per_frame[i] = site_descriptors(atoms, model)
        end
    end

    println("$t seconds")

    all_descs = Float64[]
    total_atoms = 0

    for (ifrm, frame_descs) in enumerate(descriptors_per_frame)
        for atom_desc in frame_descs
            append!(all_descs, atom_desc)
            total_atoms += 1
        end
    end

    n_features = length(descriptors_per_frame[1][1])

    desc_matrix = reshape(all_descs, (n_features, total_atoms))'
    println(size(desc_matrix))

    save_path = "/home/grethel/dev/quests/npy_files/$(output)_$(dataset).npy"
    println("Saving to $(save_path)")
    npzwrite(save_path, desc_matrix)

end
