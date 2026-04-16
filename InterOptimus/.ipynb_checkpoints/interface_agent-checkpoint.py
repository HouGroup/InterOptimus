#!/usr/bin/env python3
"""
InterOptimus Interface Generation Agent

An intelligent agent that interprets natural language descriptions to automatically
generate crystal interfaces using the InterOptimus framework. This agent can handle
various interface types including epitaxial interfaces, grain boundaries, and
heterostructures with different configurations (with/without vacuum).
"""

import re
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from InterOptimus.itworker import InterfaceWorker
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np


@dataclass
class InterfaceRequest:
    """Parsed interface generation request from natural language"""
    materials: List[str]  # [film_material, substrate_material]
    orientations: List[Tuple[int, int, int]]  # [(film_hkl), (substrate_hkl)]
    interface_type: str  # 'epitaxial', 'grain_boundary', 'heterostructure'
    vacuum_type: str  # 'with_vacuum', 'without_vacuum', 'auto'
    optimization_level: str  # 'basic', 'advanced', 'comprehensive'
    constraints: Dict[str, Any]  # Additional constraints and parameters
    structures: Optional[List[Structure]] = None  # Optional: user-provided Structure objects

    def __post_init__(self):
        if self.vacuum_type == 'auto':
            # Auto-determine vacuum based on interface type
            if self.interface_type == 'grain_boundary':
                self.vacuum_type = 'without_vacuum'
            else:
                self.vacuum_type = 'with_vacuum'


class InterfaceGenerationAgent:
    """
    Intelligent agent for automatic crystal interface generation from natural language descriptions.

    This agent can interpret various natural language inputs and map them to appropriate
    InterOptimus workflow parameters for generating crystal interfaces.
    """

    def __init__(self, structures: Optional[List[Structure]] = None):
        """
        Initialize the Interface Generation Agent.

        Args:
            structures: Optional list of pymatgen Structure objects [film, substrate].
                        If provided, these will be used instead of loading from files.
        """
        self.material_database = self._load_material_database()
        self.parameter_presets = self._load_parameter_presets()
        self.user_structures = structures

    def _load_material_database(self) -> Dict[str, str]:
        """Load common material name to formula mappings"""
        return {
            # Common materials
            'silicon': 'Si',
            'germanium': 'Ge',
            'gallium arsenide': 'GaAs',
            'aluminum oxide': 'Al2O3',
            'sapphire': 'Al2O3',
            'magnesium oxide': 'MgO',
            'titanium dioxide': 'TiO2',
            'rutile': 'TiO2',
            'zinc oxide': 'ZnO',
            'silicon dioxide': 'SiO2',
            'quartz': 'SiO2',
            'silica': 'SiO2',
            'nickel': 'Ni',
            'copper': 'Cu',
            'gold': 'Au',
            'platinum': 'Pt',
            'aluminum': 'Al',
            'titanium': 'Ti',
            'tungsten': 'W',
            'molybdenum': 'Mo',
            'tantalum': 'Ta',
            'niobium': 'Nb',
            'graphene': 'C',
            'diamond': 'C',
            'boron nitride': 'BN',
            'hexagonal boron nitride': 'BN',
            'molybdenum disulfide': 'MoS2',
            'tungsten disulfide': 'WS2',
            'graphite': 'C',
            # Add more materials as needed
        }

    def _load_parameter_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load parameter presets for heterostructure interfaces with user-defined matching criteria"""
        return {
            # Basic heterostructure settings - allows flexible matching
            'heterostructure_with_vacuum_basic': {
                'max_area': 100, 'max_length_tol': 0.05, 'max_angle_tol': 0.05,
                'film_max_miller': 2, 'substrate_max_miller': 2,
                'termination_ftol': 0.2, 'film_thickness': 15, 'substrate_thickness': 15,
                'vacuum_over_film': 12, 'double_interface': False,
            },
            'heterostructure_with_vacuum_advanced': {
                'max_area': 200, 'max_length_tol': 0.03, 'max_angle_tol': 0.03,
                'film_max_miller': 3, 'substrate_max_miller': 3,
                'termination_ftol': 0.15, 'film_thickness': 20, 'substrate_thickness': 20,
                'vacuum_over_film': 15, 'double_interface': False,
            },
            'heterostructure_without_vacuum': {
                'max_area': 150, 'max_length_tol': 0.05, 'max_angle_tol': 0.05,
                'film_max_miller': 2, 'substrate_max_miller': 2,
                'termination_ftol': 0.2, 'film_thickness': 12, 'substrate_thickness': 12,
                'vacuum_over_film': 0, 'double_interface': False,
            },
        }

    def parse_request(self, text: str) -> InterfaceRequest:
        """
        Parse natural language description into structured interface request.

        Args:
            text: Natural language description of desired interface

        Returns:
            InterfaceRequest: Structured request object
        """
        text = text.lower().strip()

        # Extract materials
        materials = self._extract_materials(text)

        # Extract orientations
        orientations = self._extract_orientations(text)

        # Determine interface type
        interface_type = self._determine_interface_type(text, materials)

        # Determine vacuum type
        vacuum_type = self._determine_vacuum_type(text)

        # Determine optimization level
        optimization_level = self._determine_optimization_level(text)

        # Extract additional constraints
        constraints = self._extract_constraints(text)

        return InterfaceRequest(
            materials=materials,
            orientations=orientations,
            interface_type=interface_type,
            vacuum_type=vacuum_type,
            optimization_level=optimization_level,
            constraints=constraints
        )

    def _extract_materials(self, text: str) -> List[str]:
        """Extract material names from text"""
        materials = []

        # Look for chemical formulas directly
        # Find all potential chemical formulas
        formula_pattern = r'\b[A-Z][a-z]*\d*(?:[A-Z][a-z]*\d*)*\b'
        all_formulas = re.findall(formula_pattern, text)

        # Filter to valid chemical formulas
        for formula in all_formulas:
            # Simple validation: chemical formulas should start with uppercase letter
            # and contain only letters and numbers
            if (len(formula) >= 1 and len(formula) <= 10 and
                formula[0].isupper() and all(c.isalnum() for c in formula)):
                # Additional check: should not be a common English word
                common_words = {'Create', 'Generate', 'Make', 'The', 'And', 'With', 'For', 'From'}
                if formula not in common_words:
                    if formula not in materials:
                        materials.append(formula)

        # Look for material names in the database
        for material_name, formula in self.material_database.items():
            if material_name in text:
                if formula not in materials:  # Avoid duplicates
                    materials.append(formula)

        # If we find "on" or "interface" patterns, try to identify film/substrate
        if len(materials) >= 2:
            # Try to determine order based on context
            if ' on ' in text:
                parts = text.split(' on ')
                if len(parts) == 2:
                    film_part = parts[0].strip()
                    substrate_part = parts[1].strip()

                    # Find materials in each part
                    film_materials = []
                    substrate_materials = []

                    for material in materials:
                        material_names = [material, self._formula_to_name(material)]
                        if any(name in film_part for name in material_names):
                            film_materials.append(material)
                        if any(name in substrate_part for name in material_names):
                            substrate_materials.append(material)

                    if film_materials and substrate_materials:
                        return film_materials[:1] + substrate_materials[:1]

        return materials[:2] if len(materials) >= 2 else materials

    def _formula_to_name(self, formula: str) -> str:
        """Convert formula back to common name"""
        for name, form in self.material_database.items():
            if form == formula:
                return name
        return formula

    def _extract_orientations(self, text: str) -> List[Tuple[int, int, int]]:
        """Extract crystal orientations from text"""
        orientations = []

        # Look for Miller indices patterns like (001), [001], 001
        miller_patterns = [
            r'\((\d+)\s*(\d+)\s*(\d+)\)',  # (001)
            r'\[(\d+)\s*(\d+)\s*(\d+)\]',  # [001]
            r'\b(\d+)\s*(\d+)\s*(\d+)\b',   # 001
        ]

        for pattern in miller_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    h, k, l = map(int, match)
                    orientation = (h, k, l)
                    if orientation not in orientations:
                        orientations.append(orientation)
                except ValueError:
                    continue

        # Default orientations if none found
        if not orientations:
            orientations = [(0, 0, 1), (0, 0, 1)]  # Default to (001) for both

        return orientations[:2]  # Return at most 2 orientations

    def _determine_interface_type(self, text: str, materials: List[str]) -> str:
        """All interfaces are treated as heterostructures with user-defined matching criteria"""
        return 'heterostructure'

    def _determine_vacuum_type(self, text: str) -> str:
        """Determine whether to include vacuum layer"""
        if 'no vacuum' in text or 'without vacuum' in text or 'vacuum-free' in text:
            return 'without_vacuum'
        elif 'with vacuum' in text or 'vacuum layer' in text:
            return 'with_vacuum'
        else:
            return 'auto'  # Let InterfaceRequest decide

    def _determine_optimization_level(self, text: str) -> str:
        """Determine optimization level"""
        if 'comprehensive' in text or 'detailed' in text or 'full' in text:
            return 'comprehensive'
        elif 'advanced' in text or 'high' in text or 'precise' in text:
            return 'advanced'
        else:
            return 'basic'

    def _extract_constraints(self, text: str) -> Dict[str, Any]:
        """Extract additional constraints from text"""
        constraints = {}

        # Extract numerical parameters
        if 'max area' in text or 'maximum area' in text:
            area_match = re.search(r'max(?:imum)?\s+area\s+(\d+)', text)
            if area_match:
                constraints['max_area'] = int(area_match.group(1))

        if 'thickness' in text:
            thickness_match = re.search(r'thickness\s+(\d+)', text)
            if thickness_match:
                constraints['thickness'] = int(thickness_match.group(1))

        return constraints

    def generate_interface(self, request: InterfaceRequest, output_dir: str = "generated_interfaces") -> Dict[str, Any]:
        """
        Generate crystal interface based on parsed request.

        Args:
            request: Parsed interface generation request
            output_dir: Directory to save generated interfaces

        Returns:
            Dictionary containing generation results and metadata
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Load or generate material structures
            structures = self._load_structures(request)

            if len(structures) < 2:
                raise ValueError("Need at least two materials to generate interface")

            # Create InterfaceWorker
            iw = InterfaceWorker(structures[0], structures[1])

            # Determine parameter preset based on vacuum type and optimization level
            level_suffix = "_advanced" if request.optimization_level == "advanced" else "_basic"
            if request.vacuum_type == "without_vacuum":
                preset_key = f"heterostructure_without_vacuum"
            else:
                preset_key = f"heterostructure_with_vacuum{level_suffix}"

            params = self.parameter_presets[preset_key].copy()

            # Apply custom constraints
            params.update(request.constraints)

            # Perform lattice matching
            print(f"Performing lattice matching for {request.materials[0]}/{request.materials[1]} interface...")
            iw.lattice_matching(
                max_area=params['max_area'],
                max_length_tol=params['max_length_tol'],
                max_angle_tol=params['max_angle_tol'],
                film_max_miller=params['film_max_miller'],
                substrate_max_miller=params['substrate_max_miller']
            )

            # Parse interface structure parameters
            iw.parse_interface_structure_params(
                termination_ftol=params['termination_ftol'],
                film_thickness=params['film_thickness'],
                substrate_thickness=params['substrate_thickness'],
                double_interface=params['double_interface'],
                vacuum_over_film=params['vacuum_over_film']
            )

            # Apply optimization if requested
            if request.optimization_level != 'basic':
                self._apply_optimization(iw, request.optimization_level)

            # Generate interface structures
            results = self._generate_structures(iw, request, output_dir)

            return {
                'success': True,
                'request': request,
                'parameters': params,
                'results': results,
                'output_dir': output_dir
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'request': request
            }

    def _load_structures(self, request: InterfaceRequest) -> List[Structure]:
        """
        Load or use crystal structures for interface generation.

        If structures are provided in the request, use them directly.
        Otherwise, fall back to loading from files.
        """
        # If structures are provided in the request, use them
        if request.structures is not None:
            if len(request.structures) >= 2:
                return request.structures[:2]
            elif len(request.structures) == 1:
                # If only one structure provided and it's a grain boundary, use it twice
                if request.interface_type == 'grain_boundary':
                    return [request.structures[0], request.structures[0]]
                else:
                    raise ValueError("Need at least two structures for interface generation")

        # Otherwise, load from materials list (legacy behavior)
        structures = []
        for material in request.materials:
            # Try to find CIF file first
            cif_path = f"{material}.cif"
            if os.path.exists(cif_path):
                try:
                    struct = Structure.from_file(cif_path)
                    structures.append(struct)
                    continue
                except Exception:
                    pass

            # For now, raise error if no CIF found
            # In a full implementation, you might query materials databases
            raise FileNotFoundError(f"Structure file for {material} not found. Please provide {material}.cif or Structure objects directly")

        return structures

    def _apply_optimization(self, iw: InterfaceWorker, level: str):
        """Apply optimization parameters based on level"""
        if level == 'advanced':
            iw.parse_optimization_params(
                set_relax_thicknesses=(4, 4),
                relax_in_layers=False,
                fmax=0.05,
                steps=200,
                device='cpu',
                do_gd=True
            )
        elif level == 'comprehensive':
            iw.parse_optimization_params(
                set_relax_thicknesses=(6, 6),
                relax_in_layers=True,
                num_relax_bayesian=20,
                fmax=0.03,
                steps=300,
                device='cpu',
                do_gd=True
            )

    def _generate_structures(self, iw: InterfaceWorker, request: InterfaceRequest,
                           output_dir: str) -> Dict[str, Any]:
        """Generate and save interface structures"""
        results = {
            'matches_found': len(iw.unique_matches),
            'structures_generated': 0,
            'files_created': []
        }

        # Generate structures for each match
        for i, match in enumerate(iw.unique_matches):
            try:
                # Get interface structure
                interface_struct = iw.get_specified_match_cib(i, ftol_by_layer_thickness=False)

                # Save structure
                filename = f"interface_{request.materials[0]}_{request.materials[1]}_match_{i}.cif"
                filepath = os.path.join(output_dir, filename)
                interface_struct.to(filename=filepath)
                results['files_created'].append(filepath)

                results['structures_generated'] += 1

            except Exception as e:
                print(f"Warning: Failed to generate structure for match {i}: {e}")
                continue

        return results

    def generate_interface_from_structures(
        self,
        film_structure: Structure,
        substrate_structure: Structure,
        vacuum_type: str = "with_vacuum",
        optimization_level: str = "basic",
        output_dir: str = "generated_interfaces",
        interface_type: str = None,  # Deprecated: kept for backward compatibility
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate heterostructure interface directly from pymatgen Structure objects.

        All interfaces are treated as heterostructures with user-defined matching criteria.
        The system allows flexible lattice matching parameters for different material combinations.

        Args:
            film_structure: pymatgen Structure object for the film material
            substrate_structure: pymatgen Structure object for the substrate material
            vacuum_type: Vacuum configuration ('with_vacuum', 'without_vacuum')
            optimization_level: Optimization level ('basic', 'advanced')
            output_dir: Directory to save generated interfaces
            interface_type: [Deprecated] Ignored - all interfaces treated as heterostructures
            **kwargs: Additional constraints and parameters:
                - max_area: Maximum interface area (default: 100-200)
                - max_length_tol: Length tolerance for matching (default: 0.03-0.05)
                - max_angle_tol: Angle tolerance for matching (default: 0.03-0.05)
                - termination_ftol: Termination fitting tolerance (default: 0.15-0.2)
                - film_thickness: Film slab thickness (default: 12-20)
                - substrate_thickness: Substrate slab thickness (default: 12-20)

        Returns:
            Dictionary containing generation results
        """
        # Parameter validation and correction
        if interface_type is not None:
            print(f"⚠️  interface_type parameter is deprecated. All interfaces are treated as heterostructures.")

        # Correct common spelling mistakes
        if vacuum_type == "without_vaccum":  # Common typo
            print("⚠️  Corrected 'without_vaccum' to 'without_vacuum'")
            vacuum_type = "without_vacuum"

        # Validate vacuum_type
        if vacuum_type not in ["with_vacuum", "without_vacuum"]:
            raise ValueError(f"vacuum_type must be 'with_vacuum' or 'without_vacuum', got '{vacuum_type}'")

        # Validate optimization_level
        if optimization_level not in ["basic", "advanced"]:
            raise ValueError(f"optimization_level must be 'basic' or 'advanced', got '{optimization_level}'")

        print(f"🔬 Generating interface from provided structures")
        print(f"📋 Film: {film_structure.composition}, Substrate: {substrate_structure.composition}")
        print(f"🎯 Vacuum: {vacuum_type}, Optimization: {optimization_level}")

        # Create a request object with the provided structures
        request = InterfaceRequest(
            materials=[film_structure.composition.reduced_formula,
                      substrate_structure.composition.reduced_formula],
            orientations=[(0, 0, 1), (0, 0, 1)],  # Default orientations
            interface_type="heterostructure",  # All interfaces are heterostructures
            vacuum_type=vacuum_type,
            optimization_level=optimization_level,
            constraints=kwargs,
            structures=[film_structure, substrate_structure]
        )

        # Generate the interface
        result = self.generate_interface(request, output_dir)

        if result['success']:
            print(f"✅ Success! Generated {result['results']['structures_generated']} interfaces")
            print(f"📁 Output: {result['output_dir']}")
        else:
            print(f"❌ Failed: {result['error']}")

        return result

    def process_text_request(self, text: str, output_dir: str = "generated_interfaces") -> Dict[str, Any]:
        """
        Main entry point: Process natural language text to generate interfaces.

        Args:
            text: Natural language description of desired interface
            output_dir: Directory to save generated interfaces

        Returns:
            Dictionary containing generation results
        """
        print(f"Processing request: {text}")

        # Parse the request
        request = self.parse_request(text)
        print(f"Parsed request: {request.materials} interface, type: {request.interface_type}, "
              f"vacuum: {request.vacuum_type}, optimization: {request.optimization_level}")

        # Generate the interface
        result = self.generate_interface(request, output_dir)

        if result['success']:
            print(f"Successfully generated {result['results']['structures_generated']} interface structures")
            print(f"Files saved to: {result['output_dir']}")
        else:
            print(f"Failed to generate interface: {result['error']}")

        return result


def main():
    """Command line interface for the Interface Generation Agent"""

    agent = InterfaceGenerationAgent()

    # Example usage
    examples = [
        "Generate a silicon on sapphire epitaxial interface with vacuum",
        "Create a grain boundary between two silicon crystals without vacuum",
        "Make a GaAs on Si heterostructure interface",
        "Generate Ni3S2 on Li2S epitaxial interface with (001) orientation",
    ]

    print("InterOptimus Interface Generation Agent")
    print("=" * 50)
    print("Examples:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")

    print("\nEnter your interface description:")
    user_input = input("> ").strip()

    if not user_input:
        print("Using default example...")
        user_input = examples[0]

    result = agent.process_text_request(user_input)

    return result


if __name__ == "__main__":
    main()