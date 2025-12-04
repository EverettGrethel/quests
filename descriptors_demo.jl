using ACEpotentials
using ExtXYZ
using Base.Threads
using NPZ  # for writing .npy / .npz

# === Model setup ===
model = ace1_model(
  elements = [:C],
  rcut = 5.5,
  order = 3,
  totaldegree = 15,
  wL = 1.5,
  transform = (:agnesi, 2, 4),
  pair_transform = (:agnesi, 1, 3),
  r0 = 2.5
)

println("Number of threads: ", nthreads())

data_file = "/home/grethel/dev/quests/examples/gap20/Graphite.xyz"
frames = collect(read_frames(data_file))

descriptors_per_frame = Vector{Vector{Vector{Float64}}}(undef, length(frames))
t = @elapsed begin
    @threads for i in eachindex(frames)
        frame = frames[i]
        atoms = Atoms(frame)
        desc = site_descriptors(atoms, model)
        descriptors_per_frame[i] = desc
    end
end

println("$t seconds")
# println("Number of frames: ", length(descriptors_per_frame))
# println("Atoms in first frame: ", length(descriptors_per_frame[1]))
# println("Features per atom: ", length(descriptors_per_frame[1][1]))

all_descs = Float64[]        # flattened descriptor data
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

# npzwrite("descriptors.npy", desc_matrix)