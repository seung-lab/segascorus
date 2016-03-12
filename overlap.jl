# T Macrina
# 160310
# 
# Given two indexed images of segmentations, calculate their overlap matrix
# 
# Use as follows on the command line:
# 	`julia overlap.jl [path to 1st segmentation] [path to 2nd segmentation]`
#

using HDF5

"""
Load indexed image with segment ids (e.g. as exported by Omni)
"""
function load_segmentation(fn)
	return h5read(fn, "main")
end

"""
Create sparse matrix counting voxels shared by segments in two segmentations

Args:
	* seg1: indexed image with segment ids
	* seg2: indexed image with segment ids

Returns:
	2D sparse matrix with ids from seg1 down one axis & ids from seg2 across the
	other. Element (i,j) represents the number of voxels shared between 
	segment #i in seg1 and segment #j in seg2.
"""
function calc_overlap_matrix(seg1, seg2)
	max_seg1, max_seg2 = maximum(seg1), maximum(seg2)
	overlap = spzeros(Int64, max_seg1+1, max_seg2+1)
	for i in 1:length(seg1)
		v = seg1[i]+1
		w = seg2[i]+1
		overlap[v, w] += 1
	end
	return overlap
end

"""
Given overlap matrix, collect information about each seg_id along the rows

Args:
	* overlap: Array{Int64, 2} (see calc_overlap_matrix output)

Returns:
	Table for each row's seg_id with following information:
		total_vx: total number of voxels contained in that segment
		max_voxel: the maximum number of voxels in a shared segment from the
			other segmentation
		seg_partners: list of the seg_ids from the other segmentation that share
			voxels with this seg_id
"""
function create_mergers_list(overlap)
	mergers = []
	for seg_id in 1:size(overlap, 1)
		voxel_counts = overlap[seg_id, :]
		total_vx = sum(voxel_counts)
		if total_vx > 0
			max_vx = maximum(voxel_counts)
			seg_partners = (rowvals(voxel_counts')-1...)
			push!(mergers, [seg_id-1, total_vx, max_vx, seg_partners])
		end
	end
	return hcat(mergers...)'
end

"""
Combine filenames of segmentations for filename of merger list
"""
function create_output_path(fn1, fn2)
	f1, f2 = splitext(splitdir(fn1)[2])[1], splitext(splitdir(fn2)[2])[1]
	fn = string(f1, "_", f2, ".txt")
	return joinpath(pwd(), fn)
end

"""
Rank mergers list for easier inspection by tracers, label table, & write to file

Segments that combine multiple segments, where the largest combined segment
represents 50% of the total, will be ranked the highest. The weighted score
multiplies that ranking by the total number of voxels. 

Consider also weighting by the number of sections merged.

Args:
	* fn: path to write the mergers table
	* list: mergers list (see output of create_mergers_list)

Returns:
	Writes tab delimited text file of ranked mergers list
"""
function write_mergers_list(fn, list)
	total_vx = convert(Array{Int64}, list[:, 2])
	max_vz = convert(Array{Int64}, list[:, 3])
	ratio = max_vz ./ max(total_vx, 1)
	rank_ratio = (1 - 2*abs(0.5-ratio))
	weighted_rank = rank_ratio .* total_vx
	list = Any[list ratio rank_ratio weighted_rank]
	list = list[sortperm(list[:, 6]; rev=true), :]
	labels = ["seg_id" "total_vx" "max_vx" "other_ids" "ratio" "rank" "weighted"]
	list = vcat(labels, list)
	writedlm(fn, list)
end

function main(fn1, fn2)
	seg1 = load_segmentation(fn1)
	seg2 = load_segmentation(fn2)
	overlap = calc_overlap_matrix(seg1, seg2)
	seg1_mergers = create_mergers_list(overlap)
	seg2_mergers = create_mergers_list(overlap')
	fn1_out = create_output_path(fn1, fn2)
	fn2_out = create_output_path(fn2, fn1)
	write_mergers_list(fn1_out, seg1_mergers)
	write_mergers_list(fn2_out, seg2_mergers)
end

if !isinteractive()
	main(ARGS[1], ARGS[2])
end