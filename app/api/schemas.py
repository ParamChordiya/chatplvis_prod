from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlotState:
    """Validated plot configuration derived from form POST."""
    sel_col: str = 'Total_Counts'
    sel_comp: str = 'All proteomes'
    plot_type: str = '2D UMAP Based'
    info_source: str = 'Function [CC]'


@dataclass(frozen=True)
class ChatRequest:
    """Validated chatbot request from the frontend."""
    node_ids: tuple[int, ...]   # immutable — use tuple, not list
    message: str
    include_similar: bool
    info_source: str


@dataclass
class ProteinNode:
    """A single protein node for the scatter plot."""
    id: int
    protein_name: str
    label: str       # HTML hover text
    x: float
    y: float
    group: str
    size: float
    color: str
    z: float | None = None

    def to_dict(self) -> dict:
        """Serialise for JSON — omit z if not set."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
