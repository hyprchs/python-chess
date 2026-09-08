SVG rendering
=============

The :mod:`chess.svg` module renders SVG Tiny 1.2 images
(mostly for IPython/Jupyter Notebook integration).
The piece images by
`Colin M.L. Burnett <https://en.wikipedia.org/wiki/User:Cburnett>`_ are triple
licensed under the GFDL, BSD and GPL.

.. autofunction:: chess.svg.piece

.. autofunction:: chess.svg.board

.. autofunction:: chess.svg.board_with_annotations

.. autoclass:: chess.svg.BoardRenderResult
    :members:

.. autoclass:: chess.svg.OverlayAnnotation
    :members:

Arrow annotations use SVG viewBox coordinates. For directed arrows, ``head_xy`` is the
painted triangular tip and ``tail_xy`` is the rear end of the painted shaft
on its centerline, not the centers of the source and destination squares.
The Lichess tail includes the round line cap and its tip includes the SVG
marker transformation. The Chess.com tail is the midpoint of the polygon's
rear edge, including the first leg of an L-shaped knight arrow.
Same-square circles retain their center points, with no arrowhead box.
Annotation changes do not change the SVG appearance.

Drag ghosts
-----------

Both board renderers accept ``ghost_squares=[chess.D4]``. Each listed square
must contain a piece. The same piece asset is rendered at 0.3 opacity, including
custom ``piece_set`` assets, while other pieces stay opaque. Ghosts form one
composited layer above board overlays; the board position and overlay annotations
are unchanged. A consumer displaying an active drag can place a separate solid
piece at the pointer position without altering the source-square ghost.

Verified by dragging the b1 knight on lichess.org/analysis on 2026-09-08,
at 100% zoom and device scale 1: ghost opacity 0.3 and z-index 2, solid dragged
piece opacity 1 and z-index 204 (the site's override of the base stylesheet).

.. autoclass:: chess.svg.Arrow
    :members:
