import sys
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import teaserpp_python


def pcd2xyz(pcd):
    return np.asarray(pcd.points).T


def extract_fpfh(pcd, voxel_size):
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T


def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1


def get_teaser_solver(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = True
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-20
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver


def merge_clouds(file1, file2, output_file, icp=0, voxel_size=0.05):
    # Load clouds
    A_pcd = o3d.io.read_point_cloud(file1)
    B_pcd = o3d.io.read_point_cloud(file2)

    for i in range(4):

        # voxel downsample both clouds
        A_pcd_downsampled = A_pcd.voxel_down_sample(voxel_size=voxel_size)
        B_pcd_downsampled = B_pcd.voxel_down_sample(voxel_size=voxel_size)

        A_xyz = pcd2xyz(A_pcd_downsampled)  # np array of size 3 by N
        B_xyz = pcd2xyz(B_pcd_downsampled)  # np array of size 3 by M

        # extract FPFH features
        print("Extracting features...")
        A_feats = extract_fpfh(A_pcd_downsampled, voxel_size)
        B_feats = extract_fpfh(B_pcd_downsampled, voxel_size)

        # establish correspondences by nearest neighbour search in feature space
        corrs_A, corrs_B = find_correspondences(
            A_feats, B_feats, mutual_filter=True)
        A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
        B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

        num_corrs = A_corr.shape[1]
        print(f'FPFH generated {num_corrs} putative correspondences.')
        if num_corrs > 10000:
            return

        print("Finding transformation using TEASER++...")
        NOISE_BOUND = voxel_size
        teaser_solver = get_teaser_solver(NOISE_BOUND)
        solution = teaser_solver.solve(A_corr, B_corr)
        print(solution)
        scale, translation, rotation = solution.scale, solution.translation, solution.rotation

        print("Transforming...")
        points = np.asarray(A_pcd.points) * scale
        transformed_points = np.dot(rotation, points.T).T + translation
        A_pcd.points = o3d.utility.Vector3dVector(transformed_points)

    # Down sample again after transformation
    A_pcd_downsampled = A_pcd.voxel_down_sample(voxel_size=voxel_size)

    # local refinement using ICP
    if icp:
        print("Running ICP...")
        icp_sol = o3d.pipelines.registration.registration_icp(
            A_pcd_downsampled, B_pcd_downsampled, NOISE_BOUND, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T_icp = icp_sol.transformation
        A_pcd = A_pcd.transform(T_icp)

    num_of_points = len(A_pcd.points) + len(B_pcd.points)
    print(num_of_points)
    # if num_of_points > 10000000

    # Merge and save
    print("Merging and saving...")
    pcd_merged = A_pcd + B_pcd
    o3d.io.write_point_cloud(output_file, pcd_merged)



def main():
    if len(sys.argv) < 5:
        print("Usage: python stitch.py <file1> <file2> <output_file> <icp>(0,1) (<voxel_size>)")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]
    icp = sys.argv[4]
	
    if len(sys.argv) >= 6:
        merge_clouds(file1, file2, output_file, int(icp), int(sys.argv[4]))
    else:
        merge_clouds(file1, file2, output_file, int(icp))


if __name__ == '__main__':
    main()
