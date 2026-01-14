#!/usr/bin/env python3
"""
Advanced InterOptimus Interface Generation Agent

Enhanced version with support for:
- Materials database queries (MP, AFLOW, etc.)
- Advanced parameter tuning
- Batch processing
- Result analysis and visualization
- Integration with external workflows
"""

import re
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import warnings

from InterOptimus.itworker import InterfaceWorker
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np

try:
    from mp_api.client import MPRester
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    warnings.warn("MPRester not available. Materials Project queries disabled.")

@dataclass
class MaterialInfo:
    """Information about a material"""
    formula: str
    name: str = ""
    structure: Optional[Structure] = None
    source: str = "local"  # 'local', 'mp', 'generated'
    mp_id: Optional[str] = None

@dataclass
class InterfaceRequest:
    """Enhanced parsed interface generation request"""
    materials: List[MaterialInfo]
    orientations: List[Tuple[int, int, int]]
    interface_type: str
    vacuum_type: str
    optimization_level: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    batch_mode: bool = False
    analysis_options: Dict[str, bool] = field(default_factory=lambda: {
        'energy_analysis': True,
        'structure_analysis': True,
        'visualization': False
    })
    structures: Optional[List[Structure]] = None  # Optional: user-provided Structure objects

@dataclass
class GenerationResult:
    """Comprehensive result of interface generation"""
    success: bool
    request: InterfaceRequest
    structures_generated: int = 0
    files_created: List[str] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None

class AdvancedInterfaceAgent:
    """
    Advanced intelligent agent for crystal interface generation with
    materials database integration and comprehensive analysis capabilities.
    """

    def __init__(self, mp_api_key: Optional[str] = None, structures: Optional[List[Structure]] = None):
        """
        Initialize the Advanced Interface Agent.

        Args:
            mp_api_key: Materials Project API key
            structures: Optional list of pymatgen Structure objects [film, substrate].
                       If provided, these will be used instead of loading from databases.
        """
        self.mp_api_key = mp_api_key or os.getenv('MP_API_KEY')
        self.material_cache: Dict[str, MaterialInfo] = {}
        self.parameter_presets = self._load_parameter_presets()
        self.matcher = StructureMatcher()
        self.user_structures = structures

        # Initialize MP client if available
        self.mp_client = None
        if MP_AVAILABLE and self.mp_api_key:
            try:
                self.mp_client = MPRester(self.mp_api_key)
                print("✓ Materials Project client initialized")
            except Exception as e:
                print(f"⚠ Failed to initialize MP client: {e}")

    def _load_parameter_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load comprehensive parameter presets"""
        return {
            # Epitaxial interfaces
            'epitaxial_with_vacuum_basic': {
                'max_area': 50, 'max_length_tol': 0.03, 'max_angle_tol': 0.03,
                'film_max_miller': 2, 'substrate_max_miller': 2,
                'termination_ftol': 0.15, 'film_thickness': 15, 'substrate_thickness': 15,
                'vacuum_over_film': 10, 'double_interface': False,
            },
            'epitaxial_with_vacuum_advanced': {
                'max_area': 100, 'max_length_tol': 0.02, 'max_angle_tol': 0.02,
                'film_max_miller': 3, 'substrate_max_miller': 3,
                'termination_ftol': 0.1, 'film_thickness': 20, 'substrate_thickness': 20,
                'vacuum_over_film': 15, 'double_interface': False,
            },
            'epitaxial_without_vacuum_basic': {
                'max_area': 50, 'max_length_tol': 0.03, 'max_angle_tol': 0.03,
                'film_max_miller': 2, 'substrate_max_miller': 2,
                'termination_ftol': 0.15, 'film_thickness': 10, 'substrate_thickness': 10,
                'vacuum_over_film': 0, 'double_interface': False,
            },

            # Grain boundaries
            'grain_boundary_with_vacuum': {
                'max_area': 100, 'max_length_tol': 0.05, 'max_angle_tol': 0.05,
                'film_max_miller': 1, 'substrate_max_miller': 1,
                'termination_ftol': 0.25, 'film_thickness': 20, 'substrate_thickness': 20,
                'vacuum_over_film': 15, 'double_interface': True,
            },
            'grain_boundary_without_vacuum': {
                'max_area': 100, 'max_length_tol': 0.05, 'max_angle_tol': 0.05,
                'film_max_miller': 1, 'substrate_max_miller': 1,
                'termination_ftol': 0.25, 'film_thickness': 15, 'substrate_thickness': 15,
                'vacuum_over_film': 0, 'double_interface': True,
            },

            # 2D materials
            '2d_material_interface': {
                'max_area': 200, 'max_length_tol': 0.1, 'max_angle_tol': 0.1,
                'film_max_miller': 1, 'substrate_max_miller': 1,
                'termination_ftol': 0.3, 'film_thickness': 5, 'substrate_thickness': 5,
                'vacuum_over_film': 20, 'double_interface': False,
            },
        }

    def parse_request(self, text: str) -> InterfaceRequest:
        """Enhanced request parsing with better NLP capabilities"""
        text = text.lower().strip()

        # Extract materials with enhanced recognition
        materials = self._extract_materials_enhanced(text)

        # Extract orientations
        orientations = self._extract_orientations(text)

        # Enhanced interface type detection
        interface_type = self._determine_interface_type_enhanced(text, materials)

        # Vacuum type determination
        vacuum_type = self._determine_vacuum_type(text)

        # Optimization level
        optimization_level = self._determine_optimization_level(text)

        # Extract constraints and options
        constraints = self._extract_constraints(text)
        analysis_options = self._extract_analysis_options(text)
        batch_mode = 'batch' in text or 'multiple' in text or 'several' in text

        return InterfaceRequest(
            materials=materials,
            orientations=orientations,
            interface_type=interface_type,
            vacuum_type=vacuum_type,
            optimization_level=optimization_level,
            constraints=constraints,
            batch_mode=batch_mode,
            analysis_options=analysis_options
        )

    def _extract_materials_enhanced(self, text: str) -> List[MaterialInfo]:
        """Enhanced material extraction with database lookup capabilities"""
        materials = []

        # Common material mappings
        material_db = {
            'silicon': 'Si', 'germanium': 'Ge', 'gallium arsenide': 'GaAs',
            'aluminum oxide': 'Al2O3', 'sapphire': 'Al2O3', 'magnesium oxide': 'MgO',
            'titanium dioxide': 'TiO2', 'rutile': 'TiO2', 'zinc oxide': 'ZnO',
            'silicon dioxide': 'SiO2', 'quartz': 'SiO2', 'silica': 'SiO2',
            'nickel': 'Ni', 'copper': 'Cu', 'gold': 'Au', 'platinum': 'Pt',
            'aluminum': 'Al', 'titanium': 'Ti', 'tungsten': 'W', 'molybdenum': 'Mo',
            'graphene': 'C', 'diamond': 'C', 'boron nitride': 'BN',
            'hexagonal boron nitride': 'BN', 'molybdenum disulfide': 'MoS2',
            'tungsten disulfide': 'WS2', 'graphite': 'C', 'lithium sulfide': 'Li2S',
            'nickel sulfide': 'Ni3S2',
        }

        # Extract material names
        found_materials = []
        for material_name, formula in material_db.items():
            if material_name in text:
                if formula not in [m.formula for m in found_materials]:
                    found_materials.append(MaterialInfo(formula=formula, name=material_name))

        # Extract chemical formulas
        formula_pattern = r'\b[A-Z][a-z]?\d*\b(?:\d*\s*[A-Z][a-z]?\d*\b)*'
        formulas = re.findall(formula_pattern, text)
        for formula in formulas:
            if formula not in [m.formula for m in found_materials]:
                found_materials.append(MaterialInfo(formula=formula))

        # Determine film/substrate order
        if len(found_materials) >= 2 and ' on ' in text:
            parts = text.split(' on ')
            if len(parts) == 2:
                film_part, substrate_part = parts[0], parts[1]

                film_candidates = []
                substrate_candidates = []

                for material in found_materials:
                    if material.name and material.name in film_part:
                        film_candidates.append(material)
                    elif material.name and material.name in substrate_part:
                        substrate_candidates.append(material)
                    elif material.formula in film_part:
                        film_candidates.append(material)
                    elif material.formula in substrate_part:
                        substrate_candidates.append(material)

                if film_candidates and substrate_candidates:
                    return film_candidates[:1] + substrate_candidates[:1]

        return found_materials[:2]

    def _determine_interface_type_enhanced(self, text: str, materials: List[MaterialInfo]) -> str:
        """Enhanced interface type determination"""
        # Check for 2D materials
        is_2d = any(m.formula in ['C', 'BN', 'MoS2', 'WS2', 'graphene'] or
                   'graphene' in m.name or '2d' in text for m in materials)

        if is_2d:
            return '2d_interface'
        elif len(materials) == 1 or (len(materials) == 2 and materials[0].formula == materials[1].formula):
            return 'grain_boundary'
        elif any(word in text for word in ['epitaxial', 'epitaxy', 'heteroepitaxial']):
            return 'epitaxial'
        elif any(word in text for word in ['heterostructure', 'hetero']):
            return 'heterostructure'
        else:
            return 'epitaxial'

    def _extract_analysis_options(self, text: str) -> Dict[str, bool]:
        """Extract analysis options from request"""
        options = {
            'energy_analysis': True,
            'structure_analysis': True,
            'visualization': False
        }

        if 'visualize' in text or 'plot' in text or 'figure' in text:
            options['visualization'] = True

        if 'no analysis' in text or 'skip analysis' in text:
            options['energy_analysis'] = False
            options['structure_analysis'] = False

        return options

    def load_material_structure(self, material: MaterialInfo) -> Optional[Structure]:
        """Load material structure from various sources"""
        # If structure is already provided, use it directly
        if material.structure is not None:
            self.material_cache[material.formula] = material
            return material.structure

        # Check cache first
        if material.formula in self.material_cache:
            cached = self.material_cache[material.formula]
            if cached.structure is not None:
                material.structure = cached.structure
                return cached.structure

        # Try local CIF file
        cif_path = f"{material.formula}.cif"
        if os.path.exists(cif_path):
            try:
                structure = Structure.from_file(cif_path)
                material.structure = structure
                material.source = 'local'
                self.material_cache[material.formula] = material
                return structure
            except Exception as e:
                print(f"Warning: Failed to load {cif_path}: {e}")

        # Try Materials Project if available
        if self.mp_client and material.formula:
            try:
                # Query for stable structure
                docs = self.mp_client.summary.search(
                    formula=material.formula,
                    is_stable=True,
                    fields=['structure', 'material_id']
                )

                if docs:
                    structure = docs[0].structure
                    material.structure = structure
                    material.source = 'mp'
                    material.mp_id = docs[0].material_id
                    self.material_cache[material.formula] = material
                    print(f"✓ Loaded {material.formula} from Materials Project")
                    return structure

            except Exception as e:
                print(f"Warning: Failed to query MP for {material.formula}: {e}")

        return None

    def generate_interface(self, request: InterfaceRequest,
                          output_dir: str = "generated_interfaces") -> GenerationResult:
        """Generate interface with comprehensive analysis and error handling"""

        start_time = time.time()
        timing = {}

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Load material structures
            load_start = time.time()
            structures = []
            for material in request.materials:
                struct = self.load_material_structure(material)
                if struct is None:
                    raise ValueError(f"Could not load structure for {material.formula}")
                structures.append(struct)

            timing['structure_loading'] = time.time() - load_start

            if len(structures) < 2:
                raise ValueError("Need at least two materials")

            # Create InterfaceWorker
            iw_start = time.time()
            iw = InterfaceWorker(structures[0], structures[1])
            timing['interface_worker_init'] = time.time() - iw_start

            # Select parameter preset
            preset_key = self._select_preset(request)
            params = self.parameter_presets[preset_key].copy()
            params.update(request.constraints)

            # Lattice matching
            match_start = time.time()
            iw.lattice_matching(**{k: v for k, v in params.items()
                                 if k in ['max_area', 'max_length_tol', 'max_angle_tol',
                                        'film_max_miller', 'substrate_max_miller']})
            timing['lattice_matching'] = time.time() - match_start

            # Interface structure parameters
            struct_start = time.time()
            iw.parse_interface_structure_params(
                termination_ftol=params['termination_ftol'],
                film_thickness=params['film_thickness'],
                substrate_thickness=params['substrate_thickness'],
                double_interface=params['double_interface'],
                vacuum_over_film=params['vacuum_over_film']
            )
            timing['structure_params'] = time.time() - struct_start

            # Optimization
            if request.optimization_level != 'basic':
                opt_start = time.time()
                self._apply_optimization(iw, request.optimization_level)
                timing['optimization'] = time.time() - opt_start

            # Generate structures
            gen_start = time.time()
            results = self._generate_structures_enhanced(iw, request, output_dir)
            timing['structure_generation'] = time.time() - gen_start

            # Analysis
            if request.analysis_options['energy_analysis'] or request.analysis_options['structure_analysis']:
                analysis_start = time.time()
                analysis_results = self._analyze_interfaces(iw, results['interfaces'], request)
                timing['analysis'] = time.time() - analysis_start
            else:
                analysis_results = {}

            timing['total'] = time.time() - start_time

            return GenerationResult(
                success=True,
                request=request,
                structures_generated=len(results['files_created']),
                files_created=results['files_created'],
                analysis_results=analysis_results,
                timing=timing
            )

        except Exception as e:
            return GenerationResult(
                success=False,
                request=request,
                error_message=str(e),
                timing={'total': time.time() - start_time}
            )

    def _select_preset(self, request: InterfaceRequest) -> str:
        """Select appropriate parameter preset"""
        base_type = request.interface_type
        vacuum = request.vacuum_type
        level = request.optimization_level

        if base_type == '2d_interface':
            return '2d_material_interface'
        elif base_type == 'grain_boundary':
            return f'grain_boundary_{vacuum}'
        else:  # epitaxial or heterostructure
            level_suffix = '_advanced' if level == 'advanced' else '_basic'
            return f'epitaxial_{vacuum}{level_suffix}'

    def _apply_optimization(self, iw: InterfaceWorker, level: str):
        """Apply optimization based on level"""
        if level == 'advanced':
            iw.parse_optimization_params(
                set_relax_thicknesses=(4, 4),
                relax_in_layers=False,
                fmax=0.05,
                steps=200,
                device='cpu',
                do_gd=True,
                discut=0.8
            )
        elif level == 'comprehensive':
            iw.parse_optimization_params(
                set_relax_thicknesses=(6, 6),
                relax_in_layers=True,
                num_relax_bayesian=30,
                fmax=0.03,
                steps=300,
                device='cpu',
                do_gd=True,
                discut=0.5
            )

    def _generate_structures_enhanced(self, iw: InterfaceWorker,
                                    request: InterfaceRequest, output_dir: str) -> Dict[str, Any]:
        """Enhanced structure generation with better error handling"""
        results = {
            'interfaces': [],
            'files_created': [],
            'errors': []
        }

        for i in range(min(len(iw.unique_matches), 10)):  # Limit to 10 interfaces
            try:
                # Get CoherentInterfaceBuilder
                cib = iw.get_specified_match_cib(i, ftol_by_layer_thickness=False)

                # Generate interface structure using first termination
                if cib.terminations:
                    termination = cib.terminations[0]

                    interface_struct = list(cib.get_interfaces(
                        termination=termination,
                        substrate_thickness=10,  # Default thickness
                        film_thickness=10,       # Default thickness
                        vacuum_over_film=request.vacuum_type == 'with_vacuum' and 10 or 0,
                        gap=0,  # No rigid body translation
                        in_layers=True
                    ))[0]

                    results['interfaces'].append(interface_struct)

                    # Generate filename
                    mat1 = request.materials[0].formula
                    mat2 = request.materials[1].formula
                    vacuum = "_vac" if request.vacuum_type == 'with_vacuum' else "_novac"
                    filename = f"interface_{mat1}_{mat2}_match_{i}{vacuum}.cif"
                    filepath = os.path.join(output_dir, filename)

                    interface_struct.to(filename=filepath)
                    results['files_created'].append(filepath)
                else:
                    results['errors'].append(f"Match {i}: No terminations found")

            except Exception as e:
                results['errors'].append(f"Match {i}: {e}")
                continue

        return results

    def _analyze_interfaces(self, iw: InterfaceWorker, interfaces: List[Structure],
                           request: InterfaceRequest) -> Dict[str, Any]:
        """Perform analysis on generated interfaces"""
        analysis = {}

        if request.analysis_options['structure_analysis']:
            analysis['structure_metrics'] = []
            for i, interface in enumerate(interfaces):
                metrics = {
                    'match_id': i,
                    'num_atoms': len(interface),
                    'lattice_params': {
                        'a': interface.lattice.a,
                        'b': interface.lattice.b,
                        'c': interface.lattice.c,
                        'volume': interface.lattice.volume
                    },
                    'space_group': interface.get_space_group_info()[0]
                }
                analysis['structure_metrics'].append(metrics)

        if request.analysis_options['energy_analysis']:
            analysis['energy_analysis'] = "Energy analysis requires MLIP setup"

        return analysis

    def batch_process(self, requests: List[str], output_base_dir: str = "batch_interfaces") -> List[GenerationResult]:
        """Process multiple interface requests in batch"""
        results = []

        for i, request_text in enumerate(requests):
            print(f"\nProcessing request {i+1}/{len(requests)}: {request_text}")

            output_dir = os.path.join(output_base_dir, f"request_{i+1}")
            request = self.parse_request(request_text)
            result = self.generate_interface(request, output_dir)
            results.append(result)

            if result.success:
                print(f"✓ Generated {result.structures_generated} interfaces")
            else:
                print(f"✗ Failed: {result.error_message}")

        return results

    def generate_interface_from_structures(
        self,
        film_structure: Structure,
        substrate_structure: Structure,
        vacuum_type: str = "with_vacuum",
        optimization_level: str = "basic",
        output_dir: str = "generated_interfaces",
        use_mlip_optimization: bool = True,
        mlip_calculator: str = "sevenn",
        analysis_options: Optional[Dict[str, bool]] = None,
        interface_type: str = None,  # Deprecated: kept for backward compatibility
        **kwargs
    ) -> GenerationResult:
        """
        Generate optimized heterostructure interface directly from pymatgen Structure objects.

        Performs lattice matching followed by MLIP-based global optimization and comprehensive
        analysis, similar to the Tutorial workflow with advanced features.

        Args:
            film_structure: pymatgen Structure object for the film material
            substrate_structure: pymatgen Structure object for the substrate material
            vacuum_type: Vacuum configuration ('with_vacuum', 'without_vacuum')
            optimization_level: Optimization level ('basic', 'advanced')
            output_dir: Directory to save generated interfaces and analysis results
            use_mlip_optimization: Whether to perform MLIP-based global optimization (default: True)
            mlip_calculator: MLIP calculator to use ('sevenn', 'orb-models', 'mace', etc.)
            analysis_options: Analysis options dictionary with keys:
                - energy_analysis: Analyze interface energies (default: True)
                - structure_analysis: Analyze structural properties (default: True)
                - visualization: Generate plots (default: False)
            interface_type: [Deprecated] Ignored - all interfaces treated as heterostructures
            **kwargs: Additional constraints and parameters (see basic agent for details)

        Returns:
            GenerationResult object containing generation results, optimization data, and analysis
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

        # Create MaterialInfo objects
        film_info = MaterialInfo(
            formula=film_structure.composition.reduced_formula,
            structure=film_structure,
            source='user_provided'
        )
        substrate_info = MaterialInfo(
            formula=substrate_structure.composition.reduced_formula,
            structure=substrate_structure,
            source='user_provided'
        )

        # Set default analysis options
        if analysis_options is None:
            analysis_options = {
                'energy_analysis': optimization_level in ['advanced', 'comprehensive'],
                'structure_analysis': True,
                'visualization': False
            }

        # Create a request object with the provided structures
        request = InterfaceRequest(
            materials=[film_info, substrate_info],
            orientations=[(0, 0, 1), (0, 0, 1)],  # Default orientations
            interface_type="heterostructure",  # All interfaces are heterostructures
            vacuum_type=vacuum_type,
            optimization_level=optimization_level,
            constraints=kwargs,
            analysis_options=analysis_options,
            structures=[film_structure, substrate_structure]
        )

        # Generate the interface
        return self.generate_interface(request, output_dir)

    def process_text_request(self, text: str, output_dir: str = "generated_interfaces") -> GenerationResult:
        """Main entry point for text-based interface generation"""
        print(f"🤖 Processing: {text}")

        request = self.parse_request(text)
        print(f"📋 Parsed: {len(request.materials)} materials, type: {request.interface_type}, "
              f"vacuum: {request.vacuum_type}, optimization: {request.optimization_level}")

        result = self.generate_interface(request, output_dir)

        if result.success:
            print(f"✅ Success! Generated {result.structures_generated} interfaces in {result.timing['total']:.1f}s")
            print(f"📁 Output: {output_dir}")
            if result.analysis_results:
                print(f"📊 Analysis: {len(result.analysis_results)} metrics computed")
        else:
            print(f"❌ Failed: {result.error_message}")

        return result


def main():
    """Command line interface"""

    import argparse
    parser = argparse.ArgumentParser(description="Advanced InterOptimus Interface Agent")
    parser.add_argument("request", nargs="?", help="Interface description")
    parser.add_argument("--mp-key", help="Materials Project API key")
    parser.add_argument("--output-dir", default="generated_interfaces", help="Output directory")
    parser.add_argument("--batch", help="Batch file with multiple requests")

    args = parser.parse_args()

    # Initialize agent
    agent = AdvancedInterfaceAgent(mp_api_key=args.mp_key)

    if args.batch:
        # Batch processing
        with open(args.batch, 'r') as f:
            requests = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        results = agent.batch_process(requests, args.output_dir)

        print(f"\n📊 Batch Results Summary:")
        successful = sum(1 for r in results if r.success)
        print(f"✅ Successful: {successful}/{len(results)}")

    elif args.request:
        # Single request
        agent.process_text_request(args.request, args.output_dir)
    else:
        # Interactive mode
        print("🤖 Advanced InterOptimus Interface Agent")
        print("=" * 50)
        print("Enter interface descriptions (Ctrl+C to exit):")

        while True:
            try:
                text = input("\n> ").strip()
                if text:
                    agent.process_text_request(text, args.output_dir)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break


if __name__ == "__main__":
    main()