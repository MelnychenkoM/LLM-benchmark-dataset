import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import re
import subprocess
from typing import Union, Optional, List
import logging

class Pocket:
    """
    Represents data for a single pocket derived from fpocket output.
    
    Args:
        pqr_file (str): Path to the PQR file containing pocket data.
        
    Attributes:
        pqr_path (str): The file path of the PQR file.
        pocket_info (dict): A dictionary containing parsed information from the HEADER section.
        alpha_spheres (numpy.ndarray): A 2D array of alpha sphere coordinates and radii.
        
    This class parses the PQR file to extract pocket-related data, including 
    sphere coordinates and radii, as well as key information from the HEADER section.
    """

    __slots__ = [
        'pqr_path', 
        'info', 
        'alpha_spheres'
    ]

    def __init__(self, pqr_file: str) -> None:
        self.pqr_path = pqr_file
        self._load_pqr_file()

    def _load_pqr_file(self):
        pattern = re.compile(r"HEADER\s+(\d+)\s+-\s+([\w\s-]+):\s+([\d.-]+)")

        self.info = {}
        self.alpha_spheres = []

        with open(self.pqr_path, "r") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                if line.startswith("HEADER"):
                    matches = pattern.findall(line)
                    if matches:
                        description = matches[0][1].strip() 
                        value = float(matches[0][2]) 
                        self.info[description] = value

                elif line.startswith("ATOM"):
                    parts = line.split()

                    x_coord = float(parts[5])
                    y_coord = float(parts[6])
                    z_coord = float(parts[7])

                    radius = float(parts[9])

                    self.alpha_spheres.append([x_coord, y_coord, z_coord, radius])

        columns = ['X', 'Y', 'Z', 'R']
        self.alpha_spheres = pd.DataFrame(self.alpha_spheres, 
                                          columns=columns, 
                                          dtype=np.float32)
    
    def __len__(self):
        return self.alpha_spheres.shape[0]

    def __getitem__(self, key):
        return self.alpha_spheres[key]
    
    def __repr__(self) -> str:
        pocket_info_str = "\n".join(
            f"{key:<30}: {value}" for key, value in self.info.items()
        )
        return (
            # f"Pocket Information (from: {self.pqr_path}):\n"
            f"{'-' * 50}\n"
            f"{pocket_info_str}\n"
            f"{'-' * 50}\n"
            f"Number of Alpha Spheres: {len(self)}"
        )
    
    def to_mol2(self, out_path: str, 
                mol_name: str ='pocket0'
                ) -> None:
        """
        Generates .mol2 file with a pocket.
        """

        sphere_template = (
            "{num_atoms:7d}  XX{x:16.4f}{y:10.4f}{z:10.4f} "
            "Du{num_atoms:7d}  UNL{num_atoms:3d}       0.0000\n"
        )

        header_lines = [
            "@<TRIPOS>MOLECULE\n",
            f"{mol_name}\n",
            "     0     0     0     0     0\n",
            "SMALL\n",
            "GASTEIGER\n\n",
            "@<TRIPOS>ATOM\n",
        ]

        coords = ""
        properties = ""
        atom_counter = 1
        num_atoms = 0

        spheres_coords = self.alpha_spheres.iloc[:, :-1].to_numpy()
        formatted_lines = np.array([
            sphere_template.format(
                num_atoms=atom_counter + num, x=row[0], y=row[1], z=row[2]
            )
            for num, row in enumerate(spheres_coords)
        ])

        coords += "".join(formatted_lines)

        for prop, value in self.info.items():
            properties += f"@<SCITEGIC>ATOM_PROPERTY\n{prop}\nSciTegic.value.StringValue\n"
            properties += "".join(
                f"{atom_counter + num} {value:.4f}\n" for num in range(len(spheres_coords))
            )

        header_lines[2] = f" {num_atoms}     0     0     0     0\n"

        with open(out_path, "w") as mol2_file:
            mol2_file.writelines(header_lines)
            mol2_file.write(coords)
            mol2_file.write(properties)


class FPocket:
    """
    Class for handling the output and execution of the fpocket tool for 
    pocket detection in protein structures.

    Args:
        pockets_folder (Optional[str]): Path to a directory containing precomputed fpocket pockets. 
                                        If provided, pockets are automatically loaded during initialization.

    Attributes:
        pockets (list): A list of Pocket objects representing detected pockets.
        info (pd.DataFrame): A DataFrame summarizing pocket information extracted from the files.
        pqr_files (list): A sorted list of .pqr files found in the pockets folder.
        min_radius (Optional[int]): Minimum radius for pocket detection.
        max_radius (Optional[int]): Maximum radius for pocket detection.

    Methods:
        run: Executes the fpocket tool on a specified PDB file and parses the results.
        __len__: Returns the number of loaded pockets.
        __getitem__: Allows indexed access to individual pockets.
    """
    def __init__(self, out_path: Optional[str] = None) -> None:
        self.min_radius = None
        self.max_radius = None
        self.out_path = out_path

        if self.out_path:
            self.out_path = Path(self.out_path)
            self.pdb_name = self.out_path.stem.split("_")[0]

            pockets_folder = self.out_path / "pockets"
            self._load_pockets(pockets_folder)

    def _load_pockets(self, pockets_folder):
        self.pockets = []
        self.info = []

        def extract_pocket_number(path):
            match = re.search(r'pocket(\d+)_vert\.pqr', path)
            return int(match.group(1)) if match else float('inf')

        pdb_path = Path(pockets_folder)
        self.pqr_files = sorted(
            [file for file in pdb_path.glob("*.pqr")],
            key=lambda file: extract_pocket_number(file.name)
        )

        for file in self.pqr_files:
            try:
                pocket = Pocket(file)
            except Exception as e:
                print(f"An error occured while loading the pocket {file}: {e}")
            self.pockets.append(pocket)
            self.info.append(pocket.info)

        self.info = pd.DataFrame(self.info)

    def __len__(self):
        if self.pockets:
            return len(self.pockets)
        raise ValueError("There are no pockets loaded. Either load them or use run method.")

    def __getitem__(self, key):
        if self.pockets:
            if isinstance(key, (list, tuple, set)):
                return [self.pockets[i] for i in key]
            return self.pockets[key]
        raise ValueError("There are no pockets loaded. Either load them or use run method.")
    
    def get_alpha_spheres(self, idx: Optional[int] = None):
        """
        Returns all alpha spheres (by default), or of a pocket of interested accessed by index.
        """
        if idx is not None:
            return self.pockets[idx].alpha_spheres

        if self.pockets:
            result = []
            for pocket in self.pockets:
                pocket_temp = pocket.alpha_spheres.copy()
                pocket_temp['Score'] = pocket.info['Drug Score']
                result.append(pocket_temp)
            return pd.concat(result, ignore_index=True)
        return None

    def run(self, 
            pdb_path: str,
            min_radius: Optional[float] = None,
            max_radius: Optional[float] = None,
            min_alpha_spheres: Optional[int] = None,
            min_clust: Optional[int] = None,
            second_cluster_dist: Optional[float] = None,
            ) -> None:
        """
        Executes fpocket on the specified PDB file with optional radius constraints.

        Args:
            pdb_path (str): Path to the PDB file to process.
            min_radius (Optional[int]): Minimum alpha sphere radius for pocket detection. Default is 3.4 A.
            max_radius (Optional[int]): Maximum alpha sphere radius for pocket detection. Default is 6.2 A.
            min_alpha_spheres (Optional[int]): This flag indicates how many alpha spheres a pocket must contain at
                                                least in order to figure in the results provided by fpocket. Default is 35.
            second_cluster_dist (Optional[int]): This parameter influences the second clustering step of small pockets
                                                to larger pockets. Default is 4.5 A.
        """
        
        command = ['fpocket']

        if min_radius:
            self.min_radius = min_radius
            command.append("-m")
            command.append(str(min_radius))

        if max_radius:
            self.max_radius = max_radius
            command.append("-M")
            command.append(str(max_radius))

        if min_alpha_spheres:
            self.min_alpha_spheres = min_alpha_spheres
            command.append("-i")
            command.append(str(min_alpha_spheres))

        if min_clust:
            self.min_clust = min_clust
            command.append("-D")
            command.append(str(min_clust))

        if second_cluster_dist:
            self.second_cluster_dist = second_cluster_dist
            command.append("-r")
            command.append(str(second_cluster_dist))

        command.append("-f")
        command.append(pdb_path)

        self.pdb_path = Path(pdb_path)
        self.pdb_name = self.pdb_path.stem
        self.out_path = self.pdb_path.parent / f"{self.pdb_path.stem}_out"

        # TODO we can use temp directory instead
        if self.out_path.exists() and self.out_path.is_dir():
            shutil.rmtree(self.out_path)

        result = subprocess.run(command, capture_output=True, text=True)
        if result.stderr:
            print(result.stderr)

        self.pockets_folder = self.out_path / "pockets"
        self._load_pockets(self.pockets_folder)

    def to_mol2(self, 
                out_path: str, 
                top_k: Optional[int] = None, 
                indicies: List[int] = None) -> None:
        """
        Generates .mol2 file with all pockets and their fpocket descriptors.
        """

        sphere_template = (
            "{num_atoms:7d}  XX{x:16.4f}{y:10.4f}{z:10.4f} "
            "Du{num_atoms:7d}  UNL{num_atoms:3d}       0.0000\n"
        )

        header_lines = [
            "@<TRIPOS>MOLECULE\n",
            f"{self.pdb_name}\n",
            "     0     0     0     0     0\n",
            "SMALL\n",
            "GASTEIGER\n\n",
            "@<TRIPOS>ATOM\n",
        ]

        coords = ""
        properties = ""
        atom_counter = 1
        num_atoms = 0

        # TODO proper error handling
        if indicies is not None:
            selected_pockets = self[indicies]
        else:
            selected_pockets = self[:top_k if top_k is not None else len(self.pockets)]

        for pocket in selected_pockets:
            spheres_coords = pocket.alpha_spheres.iloc[:, :-1].to_numpy()
            formatted_lines = np.array([
                sphere_template.format(
                    num_atoms=atom_counter + num, x=row[0], y=row[1], z=row[2]
                )
                for num, row in enumerate(spheres_coords)
            ])

            coords += "".join(formatted_lines)
            num_atoms += len(spheres_coords)

            for prop, value in pocket.info.items():
                properties += f"@<SCITEGIC>ATOM_PROPERTY\n{prop}\nSciTegic.value.StringValue\n"
                properties += "".join(
                    f"{atom_counter + num} {value:.4f}\n" for num in range(len(spheres_coords))
                )
            atom_counter += len(spheres_coords)

        header_lines[2] = f" {num_atoms}     0     0     0     0\n"

        with open(out_path, "w") as mol2_file:
            mol2_file.writelines(header_lines)
            mol2_file.write(coords)
            mol2_file.write(properties)


def save_to_mol2(out_path: str,
                 points: np.array,
                 mol_name: str = "GRID POINTS"
                 ):
    """
    Generates .mol2 file from the provided points.
    """

    sphere_template = (
        "{num_atoms:7d}  XX{x:16.4f}{y:10.4f}{z:10.4f} "
        "Du{num_atoms:7d}  UNL{num_atoms:3d}       0.0000\n"
    )

    header_lines = [
        "@<TRIPOS>MOLECULE\n",
        f"{mol_name}\n",
        "     0     0     0     0     0\n",
        "SMALL\n",
        "GASTEIGER\n\n",
        "@<TRIPOS>ATOM\n",
    ]

    coords = ""
    properties = ""
    atom_counter = 1
    num_atoms = 0

    formatted_lines = np.array([
        sphere_template.format(
            num_atoms=atom_counter + num, x=row[0], y=row[1], z=row[2]
        )
        for num, row in enumerate(points)
    ])

    coords += "".join(formatted_lines)
    num_atoms += len(points)

    header_lines[2] = f" {num_atoms}     0     0     0     0\n"

    with open(out_path, "w") as mol2_file:
        mol2_file.writelines(header_lines)
        mol2_file.write(coords)
        mol2_file.write(properties)


def save_to_pqr(
    output_pqr_path: str,
    fpocket_obj: FPocket,
    pocket_indices: Optional[List[int]] = None
) -> None:
    """
    Saves alpha spheres from selected pockets of an FPocket object
    into a single PQR file in the specified format.
    """
    if not fpocket_obj.pockets:
        print("Warning: No pockets found in the FPocket object. Output file will only contain headers.")

    header1 = "HEADER This is a pqr format file writen by the programm fpocket."
    header2 = "HEADER It contains all the pocket vertices found by fpocket."

    atom_format = ("ATOM  {serial:5d} {atom_name:^4s}{altLoc:1s}{res_name:>3s} {chain_id:1s}{pocket_idx:4d}{icode:1s}   "
                   "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{radius:6.2f}          {seg_id:<4s}\n")


    global_atom_serial_counter = 1

    try:
        with open(output_pqr_path, 'w') as outfile:
            outfile.write(header1 + "\n")
            outfile.write(header2 + "\n")

            indices_to_process = []
            if pocket_indices is None:
                indices_to_process = list(range(len(fpocket_obj)))
            else:
                for idx in pocket_indices:
                    if 0 <= idx < len(fpocket_obj):
                        indices_to_process.append(idx)
                    else:
                        print(f"Warning: Requested pocket index {idx} is out of range (0-{len(fpocket_obj)-1}). Skipping.")

            if not indices_to_process:
                 print("Warning: No valid pockets selected to write.")

            for i, pocket_idx in enumerate(indices_to_process):
                pocket = fpocket_obj[pocket_idx]
                spheres_df = pocket.alpha_spheres

                if spheres_df.empty:
                    continue

                for _, sphere_row in spheres_df.iterrows():
                    x = sphere_row['X']
                    y = sphere_row['Y']
                    z = sphere_row['Z']
                    radius = sphere_row['R']

                    outfile.write(atom_format.format(
                        serial=global_atom_serial_counter,
                        atom_name='C', 
                        altLoc=' ',        
                        res_name='STP',     
                        chain_id=' ',      
                        pocket_idx=i, 
                        icode=' ',        
                        x=x,              
                        y=y,               
                        z=z,                
                        occ=0.00,          
                        radius=radius,   
                        seg_id=''
                    ))
                    global_atom_serial_counter += 1

    except IOError as e:
        logging.warning(f"Error writing PQR file to {output_pqr_path}: {e}")
        return
    except Exception as e:
        logging.warning(f"An unexpected error occurred during PQR file writing: {e}")
        return