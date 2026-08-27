"""Shared visual contract.

One colour per concept, held across every figure in the document. Import from
here rather than redefining locally — this is what makes the figure set read as
one paper instead of a folder of plots.
"""

from __future__ import annotations

from campus_svi.analysis import paperstyle as ps

# Sources. Google is the focal series: it is the proprietary baseline the paper
# argues against, and brick red survives greyscale and colour-vision deficiency.
SOURCE_COLORS = {
    "mapillary": ps.MUTED[0],
    "google": ps.FOCAL,
    "either": "#6f6f6f",
}
SOURCE_LABELS = {
    "mapillary": "Mapillary",
    "google": "Google",
    "either": "Either source",
}

# Agreement classes, ordered light to dark so the ramp survives greyscale.
AGREEMENT_ORDER = ["neither", "mapillary_only", "google_only", "both"]
AGREEMENT_COLORS = {
    "neither": "#ececec",
    "mapillary_only": ps.MUTED[0],
    "google_only": ps.FOCAL,
    "both": "#3a3a3a",
}
AGREEMENT_LABELS = {
    "neither": "Neither",
    "mapillary_only": "Mapillary only",
    "google_only": "Google only",
    "both": "Both",
}

# Google capture programmes. `scout` is the analytically important one — it is
# trekker/tripod capture and is NOT snapped to roads — so it gets the accent.
PROGRAMME_ORDER = ["launch", "scout", "innerspace", "third_party"]
PROGRAMME_COLORS = {
    "launch": "#b8b8b8",
    "scout": "#3f8f7d",
    "innerspace": "#7f7f7f",
    "third_party": ps.FOCAL,
}
PROGRAMME_LABELS = {
    "launch": "Car (road-snapped)",
    "scout": "Trekker/tripod",
    "innerspace": "Business View",
    "third_party": "User upload",
}

# LISA cluster quadrants. Red for coverage clusters, blue for gap clusters,
# pale for the outliers, near-white for cells with no signal — so the eye goes
# to the clusters and "not significant" reads as absence rather than a
# category competing for attention.
QUADRANT_ORDER = ["HH", "LL", "LH", "HL", "ns"]
QUADRANT_COLORS = {
    "HH": "#8c2f2f",
    "LL": "#31688e",
    "LH": "#9ecae1",
    "HL": "#e6a3a3",
    "ns": "#efefef",
}
QUADRANT_LABELS = {
    "HH": "High-High",
    "LL": "Low-Low",
    "LH": "Low-High",
    "HL": "High-Low",
    "ns": "Not significant",
}

CMAP = ps.CMAP_SEQ        # one sequential ramp everywhere
CMAP_DIV = ps.CMAP_DIV    # diverging only where zero is a real midpoint
