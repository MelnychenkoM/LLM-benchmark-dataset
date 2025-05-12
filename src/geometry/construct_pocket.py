import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from biotite.structure.info import vdw_radius_protor
from scipy.spatial import ConvexHull, Delaunay

from .utils import get_residues_from_atom_array
from .fpocket import FPocket, Pocket

from typing import Union, List

def calculate_residue_atoms(protein):
    """
    Creates an array of residue centroids, with one centroid entry
    for each atom in the input AtomArray, corresponding to the
    centroid of the residue the atom belongs to.
    """
    n_atoms = len(protein)
    atom_nums = np.zeros(n_atoms, dtype=np.int16)

    unique_residues = np.unique(
        np.stack((protein.chain_id, protein.res_id.astype(str), protein.res_name)), axis=1
    ).T

    residue_len_map = {}
    for chain, res_id, res_name in unique_residues:
        residue_mask = (protein.chain_id == chain) & (protein.res_id == int(res_id)) & (protein.res_name == res_name)
        residue_atoms = protein[residue_mask]
        if len(residue_atoms) > 0:
            num_atoms = len(residue_atoms)
            residue_len_map[(chain, int(res_id), res_name)] = num_atoms
    return residue_len_map


def calculate_residue_centroids(protein: AtomArray):
    """
    Creates an array of residue centroids, with one centroid entry
    for each atom in the input AtomArray, corresponding to the
    centroid of the residue the atom belongs to.
    """
    n_atoms = len(protein)
    atom_centroids = np.zeros((n_atoms, 3), dtype=np.float32)

    unique_residues = np.unique(
        np.stack((protein.chain_id, protein.res_id.astype(str), protein.res_name)), axis=1
    ).T

    residue_centroid_map = {}
    for chain, res_id, res_name in unique_residues:
        residue_mask = (protein.chain_id == chain) & (protein.res_id == int(res_id)) & (protein.res_name == res_name)
        residue_atoms = protein[residue_mask]
        if len(residue_atoms) > 0:
            centroid = np.mean(residue_atoms.coord, axis=0)
            residue_centroid_map[(chain, int(res_id), res_name)] = centroid

    for i in range(n_atoms):
        chain = protein.chain_id[i]
        res_id = protein.res_id[i]
        res_name = protein.res_name[i]
        residue_key = (chain, res_id, res_name)
        if residue_key in residue_centroid_map:
            atom_centroids[i] = residue_centroid_map[residue_key]
        else:
            atom_centroids[i] = np.nan 
    return atom_centroids


def get_contact_atoms(pocket_atom_array: AtomArray, 
                      alpha_spheres: Union[pd.DataFrame, np.array],
                      contact_type: str = 'atoms'):
    
    contact_types = ['atoms', 'centroid']
    if contact_type not in contact_types:
        raise ValueError("Concat type should be: atoms | centroid")

    pocket_atom_coords = pocket_atom_array.coord

    if isinstance(alpha_spheres, pd.DataFrame):
        sphere_centers = alpha_spheres[['X', 'Y', 'Z']].to_numpy()
        sphere_radii = alpha_spheres['R'].to_numpy()
    elif isinstance(alpha_spheres, np.ndarray):
        sphere_centers = alpha_spheres[:, :-1]
        sphere_radii = alpha_spheres[:, -1]
    
    distances = np.linalg.norm(pocket_atom_coords[:, np.newaxis, :] - \
                               sphere_centers[np.newaxis, :, :], axis=2)
    contact_mask = distances <= sphere_radii[np.newaxis, :]
    contact_indices = np.any(contact_mask, axis=1)

    if contact_type == 'atoms':
        return pocket_atom_array[contact_indices]
    
     # DOES NOT WORK
    elif contact_type == 'centroid':
        centroids = calculate_residue_centroids(pocket_atom_array)
        distances = np.linalg.norm(centroids[:, np.newaxis, :] - \
                                   sphere_centers[np.newaxis, :, :], axis=2)

        contact_mask = distances <= sphere_radii[np.newaxis, :]
        contact_indices = np.any(contact_mask, axis=1)

        return pocket_atom_array[contact_indices]
    return


def get_contact_atoms_coverage(pocket_atom_array: AtomArray, 
                               alpha_spheres: pd.DataFrame,
                               residue_len_map: np.array,
                               threshold: float = 0.5
                               ):

    pocket_atom_coords = pocket_atom_array.coord

    sphere_centers = alpha_spheres[['X', 'Y', 'Z']].to_numpy()
    sphere_radii = alpha_spheres['R'].to_numpy()
    
    distances = np.linalg.norm(pocket_atom_coords[:, np.newaxis, :] - \
                               sphere_centers[np.newaxis, :, :], axis=2)
    contact_mask = distances <= sphere_radii[np.newaxis, :]
    contact_indices = np.any(contact_mask, axis=1)

    atoms = pocket_atom_array[contact_indices]

    for chain, res_name, res_id in zip(atoms.chain_id, atoms.res_name, atoms.res_id):
        mask = (
            (atoms.chain_id == chain) &
            (atoms.res_id == res_id) &
            (atoms.res_name == res_name)
        )
        num_atoms_residue = residue_len_map.get((chain, res_id, res_name))
        if atoms[mask] and num_atoms_residue:
            num_atoms_contact = len(atoms[mask])
            if (num_atoms_contact / num_atoms_residue) > threshold:
                return atoms


def get_concat_pockets(atom_array: AtomArray,
                       pocket_atom_array: AtomArray, 
                       fpocket: FPocket,
                       ):
    concat_pockets = []
    residue_len_map = calculate_residue_atoms(atom_array)
    for pocket in fpocket:
        alpha_spheres = pocket.alpha_spheres
        atoms = get_contact_atoms_coverage(pocket_atom_array, 
                                           alpha_spheres, 
                                           residue_len_map
                                           )
        if atoms:
            concat_pockets.append(pocket)
    return concat_pockets


def match_pockets_llm_fpocket(atom_array: AtomArray, 
                              llm_pockets: dict, 
                              fpocket: FPocket, 
                              threshold: float = 0.3
                              ) -> np.array:
    num_fpockets = len(fpocket)
    num_llm_pockets = len(llm_pockets)

    jaccard_matrix = np.zeros([num_fpockets, num_llm_pockets])
    residue_len_map = calculate_residue_atoms(atom_array)

    for i, pred_pocket in enumerate(fpocket):
        for j, llm_pocket in enumerate(llm_pockets):
            llm_pocket_atom_list = llm_pocket.get('atoms', [])
            llm_pocket_res = get_residues_from_atom_array(llm_pocket_atom_list)
            llm_res_set = set(llm_pocket_res)

            contact_atoms_in_pred = get_contact_atoms_coverage(llm_pocket_atom_list, 
                                                               pred_pocket, 
                                                               residue_len_map, 
                                                               threshold
                                                               )
            if contact_atoms_in_pred:
                pred_pocket_res_via_contact = get_residues_from_atom_array(contact_atoms_in_pred)
                pred_res_set_via_contact = set(pred_pocket_res_via_contact)

                intersection = llm_res_set.intersection(pred_res_set_via_contact)
                len_intersection = len(intersection)

                union = pred_res_set_via_contact.union(llm_res_set)
                len_union = len(union)

                jaccard_index = len_intersection / len_union if len_union > 0 else 0.0
                jaccard_matrix[i, j] = jaccard_index
    return jaccard_matrix


def combine_pockets(pockets, indices_to_combine: np.ndarray):
    """ Combines pockets at given indices. """
    if not indices_to_combine:
        return []
    selected_pockets = [pockets[i] for i in indices_to_combine]
    if not selected_pockets:
        return []
    combined_name_parts = [p['pocket_id'] for p in selected_pockets]
    
    combined_atoms = sum((p['atoms'] for p in selected_pockets), AtomArray(0))
    combined_scores = [p['score'] for p in selected_pockets]
    return [{
        'pocket_id': '_'.join(combined_name_parts),
        'score': np.mean(combined_scores) if combined_scores else 0.0,
        'atoms': combined_atoms
    }]


def get_pocket_to_merge(jaccard_matrix: np.array, threshold: float):
    if jaccard_matrix.shape[1] > 1:
        for i, row in enumerate(jaccard_matrix):
            indices_to_merge_bool = row > threshold
            indices_to_merge = np.where(indices_to_merge_bool)[0].tolist()
            if len(indices_to_merge) >= 2:
                return indices_to_merge
    return []


def recursive_merge_and_match(atom_array: AtomArray, 
                              llm_pockets: dict, 
                              fpocket: FPocket, 
                              threshold_merge=0.6, 
                              threshold_residue=0.3
                              ):
    jaccard_matrix = match_pockets_llm_fpocket(atom_array, 
                                               llm_pockets , 
                                               fpocket, 
                                               threshold=threshold_residue
                                               )

    indicies_to_merge = get_pocket_to_merge(jaccard_matrix, threshold=threshold_merge)

    if not indicies_to_merge:
        return llm_pockets, jaccard_matrix
        
    merged_pocket = combine_pockets(llm_pockets, indicies_to_merge)
    excluded_pockets = [llm_pockets[i] for i in range(len(llm_pockets)) \
                        if i not in indicies_to_merge]
    
    new_pockets = merged_pocket + excluded_pockets

    return recursive_merge_and_match(
        atom_array, 
        new_pockets,
        fpocket, 
        threshold_merge, 
        threshold_residue
    )


def get_alpha_spheres(pocket_list: List[Pocket]):
    result = []
    for pocket in pocket_list:
        result.append(pocket.alpha_spheres)
    return pd.concat(result)


def get_contact_residues_mask(protein: AtomArray, atoms: AtomArray):
    mask = np.any(
        (protein.chain_id[:, np.newaxis] == atoms.chain_id) &
        (protein.res_name[:, np.newaxis] == atoms.res_name) &
        (protein.res_id[:, np.newaxis] == atoms.res_id),
        axis=1
    )
    return mask


def get_pocket_match_indicies(jaccard_matrix: np.ndarray, 
                              threshold: float = 0.1):
    """ Finds matching column indices in a Jaccard matrix based on a threshold. """
    threshold_array = np.ones(jaccard_matrix.shape[1]) * threshold

    assigned_column = np.full(jaccard_matrix.shape[0], -1, dtype=int)

    for r in range(jaccard_matrix.shape[0]):
        row = jaccard_matrix[r]
        valid_indices = np.where(row >= threshold_array)[0]

        if valid_indices.size > 0:
            best_index = valid_indices[np.argmax(row[valid_indices])]
            assigned_column[r] = best_index
    return assigned_column


def get_pocket_match_mask(jaccard_matrix: np.ndarray, 
                          threshold: float = 0.1,
                          return_best_match: bool = True
                          ):
    """ Finds matching column indices in a Jaccard matrix based on a threshold."""

    num_rows, num_columns = jaccard_matrix.shape
    mask = np.zeros((num_columns, num_rows), dtype=bool)

    if return_best_match:
        if num_rows > 0:
            best_col_indices = np.argmax(jaccard_matrix, axis=1)
            mask[best_col_indices, np.arange(num_rows)] = (
                jaccard_matrix[np.arange(num_rows), best_col_indices] >= threshold
            )
    else:  
        mask = jaccard_matrix.T >= threshold
    return mask


def create_grid(alpha_spheres: np.array, 
                spacing: float = 2
                ) -> np.array:
    """ Creates a grid with an equal spacing for the provided alpha spheres. """
    centers = alpha_spheres[:, :-1]
    radii = alpha_spheres[:, -1]

    min_sphere_extents = centers - radii[:, np.newaxis]
    max_sphere_extents = centers + radii[:, np.newaxis]

    min_coords = np.min(min_sphere_extents, axis=0)
    max_coords = np.max(max_sphere_extents, axis=0) 

    x_range = np.arange(min_coords[0], max_coords[0], spacing)
    y_range = np.arange(min_coords[1], max_coords[1], spacing)
    z_range = np.arange(min_coords[2], max_coords[2], spacing)

    xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    distances = np.linalg.norm(grid_points[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    contact_mask = distances <= radii[np.newaxis, :]
    
    contact_indices = np.any(contact_mask, axis=1)
    return grid_points[contact_indices]


def correct_volume_grid(grid: np.ndarray, 
                        atom_array: AtomArray, 
                        alpha_spheres: np.ndarray, 
                        ):
    """ Corrects grid using the convex hull on pocket atoms. """
    atoms = get_contact_atoms(atom_array, alpha_spheres)
    mask = get_contact_residues_mask(atom_array, atoms)

    residue_coords = atom_array[mask].coord
    
    if residue_coords.shape[0] < 4:
        return grid

    hull = ConvexHull(residue_coords)
    delaunay = Delaunay(residue_coords[hull.vertices])
    inside_points_mask = delaunay.find_simplex(grid) >= 0
    inside_points = grid[inside_points_mask]

    return inside_points


def clean_grid(grid: np.ndarray, 
               protein: AtomArray, 
               radius: float = 1.5
                ):
    """ Cleans grid points too close to some atom. """
    atom_coord = protein.coord
    
    distances = np.linalg.norm(grid[:, np.newaxis, :] - atom_coord[np.newaxis, :, :], axis=2)

    if isinstance(radius, (int, float)):
        contact_mask = distances <= radius
    elif radius == 'auto':
        radius_vdw = [
            vdw_radius_protor(atom.res_name, atom.atom_name)
            for atom in protein
        ]
        contact_mask = distances <= radius_vdw
    else:
        raise ValueError(f"The parameter 'radius' should be a number or 'auto'.")
    contact_indices = np.any(contact_mask, axis=1)

    return grid[~contact_indices]


class PocketConstructor():
    def __init__(self):
        self.residues = None
        self.hull = None
        self.delaunay = None

    def correct_volume_grid(self,
                            grid: np.ndarray,
                            atom_array: AtomArray, 
                            alpha_spheres: np.ndarray
                            ) -> np.ndarray:
        self.atoms = get_contact_atoms(atom_array, alpha_spheres)
        self.mask = get_contact_residues_mask(atom_array, self.atoms)

        self.residues = atom_array[self.mask]
        self.residue_coords = self.residues.coord
        
        if self.residue_coords.shape[0] < 4:
            return grid

        self.hull = ConvexHull(self.residue_coords)
        self.delaunay = Delaunay(self.residue_coords[self.hull.vertices])
        inside_points_mask = self.delaunay.find_simplex(grid) >= 0
        inside_points = grid[inside_points_mask]

        return inside_points
    
    @staticmethod
    def clean_grid(grid: np.ndarray, 
                   protein: AtomArray, 
                   radius: Union[float, str] = 1.5
                   ):
        """ Cleans grid points too close to the atoms. """
        atom_coord = protein.coord
        
        distances = np.linalg.norm(grid[:, np.newaxis, :] - atom_coord[np.newaxis, :, :], axis=2)

        if isinstance(radius, (int, float)):
            contact_mask = distances <= radius
        elif radius == 'auto':
            radius_vdw = [
                vdw_radius_protor(atom.res_name, atom.atom_name) if 
                vdw_radius_protor(atom.res_name, atom.atom_name) is not None else 1.5
                for atom in protein
            ]
            contact_mask = distances <= radius_vdw
        else:
            raise ValueError(f"The parameter 'radius' should be float or 'auto'.")
        contact_indices = np.any(contact_mask, axis=1)
        return grid[~contact_indices]
    
    def make_tcl_pocket(self,
                        path: str,
                        vertex_color: str = "blue",
                        vertex_radius: float = 0.3,
                        edge_color: str = "blue",
                        edge_width: int = 2,
                        face_color: str = "gray",
                        face_material: str = "Transparent"
                        ):
        """
        Generates a VMD Tcl script to visualize the convex hull.

        This method creates a .tcl file that, when sourced in VMD,
        draws the vertices of the convex hull (calculated in
        correct_volume_grid) as spheres and the edges connecting them
        as cylinders.
        """
        if self.hull is None or self.residue_coords is None:
            raise ValueError("Convex hull data not available. "
                             "Run 'correct_volume_grid' successfully first.")

        tcl_commands = []
        tcl_commands.append("# VMD Tcl Script for Convex Hull Visualization")
        tcl_commands.append("# Generated by PocketConstructor")

        tcl_commands.append(f"set vertex_radius {vertex_radius}")
        tcl_commands.append(f"set edge_width {edge_width}")

        tcl_commands.append("\n# --- Script Start ---")
        tcl_commands.append("# Draw Convex Hull Vertices")
        tcl_commands.append(f"draw color {vertex_color}")

        hull_vertex_coords = self.residue_coords[self.hull.vertices]
        for i, coord in enumerate(hull_vertex_coords):
            coord_str = f"{{{coord[0]:.4f} {coord[1]:.4f} {coord[2]:.4f}}}"
            tcl_commands.append(
                # f"draw sphere {coord_str} radius {vertex_radius}"
                f"graphics sphere {coord_str} radius $vertex_radius resolution 80"
            )
        tcl_commands.append("")

        tcl_commands.append("# Draw Convex Hull Edges")
        tcl_commands.append(f"draw color {edge_color}")

        edges = set()
        for simplex in self.hull.simplices:
            edge1 = tuple(sorted((simplex[0], simplex[1])))
            edge2 = tuple(sorted((simplex[1], simplex[2])))
            edge3 = tuple(sorted((simplex[2], simplex[0])))
            edges.add(edge1)
            edges.add(edge2)
            edges.add(edge3)

        for idx1, idx2 in edges:
            coord1 = self.residue_coords[idx1]
            coord2 = self.residue_coords[idx2]

            coord1_str = f"{{{coord1[0]:.4f} {coord1[1]:.4f} {coord1[2]:.4f}}}"
            coord2_str = f"{{{coord2[0]:.4f} {coord2[1]:.4f} {coord2[2]:.4f}}}"
            tcl_commands.append(
                # f"draw line {coord1_str} {coord2_str}"
                f"graphics line {coord1_str} {coord2_str} width $edge_width style solid"
            )
        tcl_commands.append("")

        tcl_commands.append("# Draw Convex Hull Faces (Triangles)")
        tcl_commands.append(f"draw color {face_color}")
        tcl_commands.append(f"draw material {face_material}")

        for simplex in self.hull.simplices:
            idx1, idx2, idx3 = simplex
            coord1 = self.residue_coords[idx1]
            coord2 = self.residue_coords[idx2]
            coord3 = self.residue_coords[idx3]

            coord1_str = f"{{{coord1[0]:.4f} {coord1[1]:.4f} {coord1[2]:.4f}}}"
            coord2_str = f"{{{coord2[0]:.4f} {coord2[1]:.4f} {coord2[2]:.4f}}}"
            coord3_str = f"{{{coord3[0]:.4f} {coord3[1]:.4f} {coord3[2]:.4f}}}"

            tcl_commands.append(
                # f"draw triangle {coord1_str} {coord2_str} {coord3_str}"
                f"graphics triangle {coord1_str} {coord2_str} {coord3_str}"
            )

        tcl_commands.append("\n# --- Script End ---")
        tcl_script = "\n".join(tcl_commands)

        try:
            with open(path, 'w') as f:
                f.write(tcl_script)
        except IOError as e:
            return