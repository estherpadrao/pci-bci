from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, Optional
import pandas as pd

from src.scenarios.artifacts import ArtifactStore, stable_hash
from src.scenarios.scenario import Scenario
from src.scenarios.apply_mass_edits import apply_mass_edits

from src.mass.topography import build_composite_topography, build_bci_masses
from src.network.build_scenario_graph import build_scenario_graph
from src.network.dijkstra import compute_hex_travel_times

from src.indices.pci import compute_pci
from src.indices.bci import compute_bci

class ScenarioRunner:
    """
    Runs scenarios in the enforced order:

      (1) hex masses + topography (clipped)
      (2) scenario graph (base graph + network edits + topo-weighting)
      (3) Dijkstra travel times
      (4) PCI and BCI

    Uses ArtifactStore to cache outputs by stable hashes.

    IMPORTANT: This runner separates MASS changes from DISTANCE changes:
      - travel_times is cached by a graph_key (distance)
      - indices are cached by (travel_times_key + mass_key + params)
    """

    def __init__(self, store: ArtifactStore, base_assets: Dict[str, Any]):
        """
        base_assets must include:
          - hex_ids: pd.Index of hex ids
          - layers: dict[str, MassLayer] baseline hex-clipped layers (amenities, parks, etc.)
          - weights: dict[str,float] baseline weights for PCI composite
          - G_base: nx.MultiDiGraph base multimodal network (with time_min)
          - node_to_hex: dict[node -> hex_id]
          - hex_to_node: dict[hex_id -> node]
          - population: pd.Series hex-indexed
          - income: pd.Series hex-indexed (from Census later)
          - labour: pd.Series hex-indexed
          - hex_area_m2: pd.Series hex-indexed (needed for park coverage mask if parks is area)
        """
        self.store = store
        self.base = base_assets

        required = ["hex_ids", "layers", "weights", "G_base", "node_to_hex", "hex_to_node",
                    "population", "income", "labour"]
        missing = [k for k in required if k not in self.base]
        if missing:
            raise ValueError(f"base_assets missing required keys: {missing}")

    # -----------------------------
    # Keys / dependency hashing
    # -----------------------------
    def _mass_key(self, scenario: Scenario) -> str:
        # mass edits + weights + topo rules define PCI topo and any derived masses
        payload = {
            "mass_edits": [e.__dict__ for e in scenario.mass_edits],
            "weights": scenario.weights,
            "topo_rules": scenario.topo_rules,
        }
        return stable_hash(payload)

    def _graph_key(self, scenario: Scenario, topo_key: str) -> str:
        # graph depends on network edits + gamma + topo_key (because topo affects edge weights)
        payload = {
            "network_edits": [e.__dict__ for e in scenario.network_edits],
            "gamma": scenario.gamma,
            "topo_key": topo_key,
        }
        return stable_hash(payload)

    def _travel_times_key(self, graph_key: str, max_time: float) -> str:
        return stable_hash({"graph_key": graph_key, "max_time": float(max_time)})

    # -----------------------------
    # Main run
    # -----------------------------
    def run(
        self,
        scenario: Scenario,
        max_time: float = 60.0,
        recompute: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        """
        recompute: optional set of stages to force recompute:
          {"mass", "graph", "travel_times", "pci", "bci"}
        """
        recompute = recompute or set()

        hex_ids: pd.Index = self.base["hex_ids"]
        base_layers = self.base["layers"]          # dict[str, MassLayer]
        base_weights = self.base["weights"]        # dict[str,float]

        # 1) MASS + TOPOGRAPHY (PCI topo + BCI masses)
        mass_key = self._mass_key(scenario)

        layers_key = stable_hash({"mass_key": mass_key, "kind": "layers_applied"})
        layers = None if ("mass" in recompute) else self.store.load("layers", layers_key)
        weights = None if ("mass" in recompute) else self.store.load("weights", layers_key)

        if layers is None or weights is None:
            layers, weights = apply_mass_edits(
                layers=base_layers,
                weights={**base_weights, **scenario.weights},
                hex_ids=hex_ids,
                mass_edits=scenario.mass_edits,
            )
            self.store.save("layers", layers_key, layers)
            self.store.save("weights", layers_key, weights)

        topo_key = stable_hash({"mass_key": mass_key, "kind": "topo_pci"})
        topo_pci = None if ("mass" in recompute) else self.store.load("topo_pci", topo_key)
        if topo_pci is None:
            topo_res = build_composite_topography(
                layers=layers,
                weights=weights,
                hex_ids=hex_ids,
                topo_rules=scenario.topo_rules,
            )
            topo_pci = topo_res.topo
            self.store.save("topo_pci", topo_key, topo_pci)

        # BCI destination masses (hex-indexed)
        pop = self.base["population"].reindex(hex_ids)
        inc = self.base["income"].reindex(hex_ids)
        lab = self.base["labour"].reindex(hex_ids)

        bci_mass_key = stable_hash({"kind": "bci_masses", "source": "base_assets"})
        bci_masses = self.store.load("bci_masses", bci_mass_key)
        if bci_masses is None:
            market_mass, labour_mass = build_bci_masses(pop, inc, lab, hex_ids)
            bci_masses = (market_mass, labour_mass)
            self.store.save("bci_masses", bci_mass_key, bci_masses)
        else:
            market_mass, labour_mass = bci_masses

        # 2) SCENARIO GRAPH (distance)
        graph_key = self._graph_key(scenario, topo_key)
        G_s = None if ("graph" in recompute) else self.store.load("G_s", graph_key)
        if G_s is None:
            G_s = build_scenario_graph(
                G_base=self.base["G_base"],
                topo_hex=topo_pci,              # topo affects edge weights by design
                node_to_hex=self.base["node_to_hex"],
                scenario=scenario
            )
            self.store.save("G_s", graph_key, G_s)

        # 3) TRAVEL TIMES (Dijkstra)
        tt_key = self._travel_times_key(graph_key, max_time)
        travel_times = None if ("travel_times" in recompute) else self.store.load("travel_times", tt_key)
        if travel_times is None:
            travel_times = compute_hex_travel_times(
                G=G_s,
                hex_to_node=self.base["hex_to_node"],
                max_time=max_time,
                progress_every=50
            )
            self.store.save("travel_times", tt_key, travel_times)

        # 4) PCI (fast if travel_times cached)
        pci_key = stable_hash({
            "tt_key": tt_key,
            "topo_key": topo_key,
            "betas": scenario.betas,
            "active_lambda": scenario.active_lambda,
            "mode_cost": scenario.mode_cost,
            "park_mask_enabled": scenario.park_mask_enabled,
            "park_mask_threshold": scenario.park_mask_threshold,
        })
        pci = None if ("pci" in recompute) else self.store.load("pci", pci_key)
        if pci is None:
            pci = compute_pci(
                travel_times=travel_times,
                topo_mass=topo_pci,
                layers=layers,
                betas=scenario.betas,
                income=inc,
                mode_cost=scenario.mode_cost,
                active_lambda=scenario.active_lambda,
                park_mask_enabled=scenario.park_mask_enabled,
                park_mask_threshold=scenario.park_mask_threshold,
                hex_area_m2=self.base.get("hex_area_m2"),
            )
            self.store.save("pci", pci_key, pci)

        # 5) BCI (fast if travel_times cached)
        bci_key = stable_hash({
            "tt_key": tt_key,
            "betas": scenario.betas,
            "combine": "weight_free",
            "interface_lambda": scenario.interface_lambda,
        })
        bci_out = None if ("bci" in recompute) else self.store.load("bci_out", bci_key)
        if bci_out is None:
            bci_out = compute_bci(
                travel_times=travel_times,
                population=pop,
                income=inc,
                labour=lab,
                betas=scenario.betas,
                weights=None,
                combine="weight_free",
                interface=self.base.get("urban_interface"),
                interface_lambda=scenario.interface_lambda,
            )
            self.store.save("bci_out", bci_key, bci_out)

        return {
            "scenario": scenario,
            "layers": layers,
            "weights": weights,
            "topo_pci": topo_pci,
            "market_mass": market_mass,
            "labour_mass": labour_mass,
            "G_s": G_s,
            "travel_times": travel_times,
            "pci": pci,
            "bci": bci_out["BCI"],
            "A_market": bci_out["A_market"],
            "A_labour": bci_out["A_labour"],
        }
