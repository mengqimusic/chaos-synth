# chaos-synth — Preset system: save/load/morph TOML parameter snapshots
# Phase 3 Task A4

import toml
import os

PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'presets')


class PresetManager:
    """Save/load/morph parameter snapshots via TOML.

    Usage:
        pm = PresetManager()
        preset = pm.capture(chaos, manifold, pool, coupling, delay_net, macros)
        pm.save(preset, "my_preset")
        loaded = pm.load("my_preset")
        morphed = pm.morph(preset_a, preset_b, t=0.5)
    """

    @staticmethod
    def capture(chaos, manifold, pool, coupling, delay_net, macros: dict) -> dict:
        """Capture current system state as a preset dict.

        Args:
            chaos: Chaos engine instance (LogisticMap, LorenzAttractor, or RoesslerAttractor).
            manifold: ManifoldMapper instance.
            pool: VoicePool instance.
            coupling: CouplingField instance.
            delay_net: DelayNetwork instance.
            macros: dict of macro control values (material, density, mutation, coherence,
                    feedback, etc.).

        Returns:
            dict with keys: name, chaos, manifold, pool, coupling, delay_net, macros.
        """
        # Detect chaos type and capture relevant attributes
        chaos_type = type(chaos).__name__
        chaos_dict = {'type': chaos_type}

        if chaos_type == 'LogisticMap':
            chaos_dict['r'] = getattr(chaos, 'r', 3.7)
        elif chaos_type == 'LorenzAttractor':
            chaos_dict['sigma'] = getattr(chaos, 'sigma', 10.0)
            chaos_dict['rho'] = getattr(chaos, 'rho', 28.0)
            chaos_dict['beta'] = getattr(chaos, 'beta', 2.667)
            chaos_dict['dt'] = getattr(chaos, 'dt', 0.01)
        elif chaos_type == 'RoesslerAttractor':
            chaos_dict['a'] = getattr(chaos, 'a', 0.2)
            chaos_dict['b'] = getattr(chaos, 'b', 0.2)
            chaos_dict['c'] = getattr(chaos, 'c', 5.7)
            chaos_dict['dt'] = getattr(chaos, 'dt', 0.03)

        preset = {
            'name': 'untitled',
            'chaos': chaos_dict,
            'manifold': {
                'n_centroids': len(manifold.centroids),
            },
            'pool': {
                'capacity': pool.capacity,
                'max_active': pool.max_active,
            },
            'macros': macros,
        }

        # Coupling and delay_net are optional / stored as presence flags
        if coupling is not None:
            preset['coupling'] = {'enabled': True}
        if delay_net is not None:
            preset['delay_net'] = {
                'wet_mix': float(getattr(delay_net, 'wet_mix', 0.3)),
            }

        return preset

    @staticmethod
    def save(preset: dict, name: str) -> str:
        """Save a preset dict to config/presets/<name>.toml.

        Args:
            preset: Preset dict (e.g. from capture()).
            name: File name without .toml extension.

        Returns:
            Full path to the saved file.
        """
        os.makedirs(PRESET_DIR, exist_ok=True)
        # Update the name field
        preset['name'] = name
        path = os.path.join(PRESET_DIR, f'{name}.toml')
        with open(path, 'w') as f:
            toml.dump(preset, f)
        return path

    @staticmethod
    def load(name: str) -> dict:
        """Load a preset from config/presets/<name>.toml.

        Args:
            name: File name without .toml extension.

        Returns:
            Preset dict.

        Raises:
            FileNotFoundError: if the preset file doesn't exist.
        """
        path = os.path.join(PRESET_DIR, f'{name}.toml')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Preset not found: {path}")
        with open(path, 'r') as f:
            return toml.load(f)

    @staticmethod
    def list_presets() -> list:
        """List available preset names (without .toml extension).

        Returns:
            List of preset name strings. Empty list if directory doesn't exist.
        """
        if not os.path.isdir(PRESET_DIR):
            return []
        return sorted([f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith('.toml')])

    @staticmethod
    def morph(preset_a: dict, preset_b: dict, t: float) -> dict:
        """Linear interpolation between two presets.

        t=0 → preset_a, t=1 → preset_b.
        Numeric values are linearly interpolated.
        String values pick A if t < 0.5 else B.

        Args:
            preset_a: First preset dict.
            preset_b: Second preset dict.
            t: Interpolation factor in [0, 1].

        Returns:
            Morphed preset dict.
        """
        t = max(0.0, min(1.0, t))  # clamp
        result = {}

        for key in preset_a:
            if key not in preset_b:
                # Key only in A; keep as-is
                result[key] = preset_a[key]
                continue

            a_val = preset_a[key]
            b_val = preset_b[key]

            if key == 'name':
                result[key] = f'morph_{preset_a["name"]}_{preset_b["name"]}'
            elif isinstance(a_val, dict) and isinstance(b_val, dict):
                result[key] = {}
                for subkey in a_val:
                    if subkey in b_val:
                        a_sub = a_val[subkey]
                        b_sub = b_val[subkey]
                        if isinstance(a_sub, (int, float)) and isinstance(b_sub, (int, float)):
                            result[key][subkey] = a_sub + (b_sub - a_sub) * t
                        else:
                            result[key][subkey] = a_sub if t < 0.5 else b_sub
                    else:
                        result[key][subkey] = a_val[subkey]
            elif isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                result[key] = a_val + (b_val - a_val) * t
            else:
                # Fallback for strings, lists, etc.
                result[key] = a_val if t < 0.5 else b_val

        # Include keys only in B
        for key in preset_b:
            if key not in result:
                result[key] = preset_b[key]

        return result
