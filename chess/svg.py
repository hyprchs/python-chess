from __future__ import annotations

import base64
import math
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chess

from typing import Dict, Iterable, Literal, Optional, Tuple, Union
from chess import Color, IntoSquareSet, Square


SQUARE_SIZE = 45

CoordinateStyle = Literal["lichess", "chess.com"]
ArrowStyle = Literal["lichess", "chess.com"]
LegalMoveStyle = Literal["lichess", "chess.com"]
CanonicalOverlayColor = Literal["green", "red", "yellow", "blue"]


@dataclass(frozen=True)
class DestinationMarker:
    """An explicitly placed destination dot or capture ring, independent of move legality."""

    square: Square
    kind: Literal["dot", "capture"]

    def __post_init__(self) -> None:
        if type(self.square) is not int or self.square not in chess.SQUARES:
            raise ValueError("destination marker square must be a chess square")
        if self.kind not in ("dot", "capture"):
            raise ValueError(f"unsupported destination marker kind: {self.kind!r}")


@dataclass(frozen=True)
class UserHighlight:
    """A Lichess-style circular user highlight with a site color palette."""

    square: Square
    color: CanonicalOverlayColor
    palette: LegalMoveStyle = "lichess"

    def __post_init__(self) -> None:
        if self.color not in {"green", "red", "yellow", "blue"}:
            raise ValueError(f"unsupported user highlight color: {self.color!r}")
        if self.palette not in {"lichess", "chess.com"}:
            raise ValueError(f"unsupported user highlight palette: {self.palette!r}")


@dataclass(frozen=True)
class OverlayAnnotation:
    """Semantic bounds for a primitive emitted by :func:`board_with_annotations`.

    Straight-arrow OBBs align with the tail-to-head axis. Chess.com knight
    arrows and same-square circles use board-aligned bounds.

    For directed arrows, ``head_xy`` is the painted triangular tip and
    ``tail_xy`` is the rear end of the painted shaft along its centerline,
    not the logical source/destination square centers. Same-square circles
    retain their center points.
    """

    kind: Literal[
        "arrow",
        "user_highlight",
        "legal_destination_dot",
        "legal_destination_capture",
    ]
    bbox_xyxy: Tuple[float, float, float, float]
    arrowhead_bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    color: Optional[CanonicalOverlayColor] = None
    tail_xy: Optional[Tuple[float, float]] = None
    head_xy: Optional[Tuple[float, float]] = None
    obb_xyxyxyxy: Optional[Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]] = None


@dataclass(frozen=True)
class BoardRenderResult:
    """The SVG board and annotations generated from its exact SVG primitives."""

    svg: "SvgWrapper"
    viewbox_size: int
    annotations: Tuple[OverlayAnnotation, ...]


@lru_cache(maxsize=1)
def available_piece_sets() -> list[str]:
    """
    Returns the available piece set names shipped in ``chess/piece``.

    Note: python-chess sets ``zip_safe=False``, so package data is available on
    the filesystem and can be listed directly.
    """
    piece_root = Path(__file__).resolve().parent / "piece"
    if not piece_root.is_dir():
        return []
    return sorted(p.name for p in piece_root.iterdir() if p.is_dir())

PIECES = {
    "b": """<g id="black-bishop" class="black bishop" fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2zm6-4c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2zM25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z" fill="#000" stroke-linecap="butt"/><path d="M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5" stroke="#fff" stroke-linejoin="miter"/></g>""",  # noqa: E501
    "k": """<g id="black-king" class="black king" fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6" stroke-linejoin="miter"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#000" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z" fill="#000"/><path d="M20 8h5" stroke-linejoin="miter"/><path d="M32 29.5s8.5-4 6.03-9.65C34.15 14 25 18 22.5 24.5l.01 2.1-.01-2.1C20 18 9.906 14 6.997 19.85c-2.497 5.65 4.853 9 4.853 9M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0" stroke="#fff"/></g>""",  # noqa: E501
    "n": """<g id="black-knight" class="black knight" fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" style="fill:#000000; stroke:#000000;"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" style="fill:#000000; stroke:#000000;"/><path d="M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z" style="fill:#ececec; stroke:#ececec;"/><path d="M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z" transform="matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)" style="fill:#ececec; stroke:#ececec;"/><path d="M 24.55,10.4 L 24.1,11.85 L 24.6,12 C 27.75,13 30.25,14.49 32.5,18.75 C 34.75,23.01 35.75,29.06 35.25,39 L 35.2,39.5 L 37.45,39.5 L 37.5,39 C 38,28.94 36.62,22.15 34.25,17.66 C 31.88,13.17 28.46,11.02 25.06,10.5 L 24.55,10.4 z " style="fill:#ececec; stroke:none;"/></g>""",  # noqa: E501
    "p": """<g id="black-pawn" class="black pawn"><path d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#000" stroke="#000" stroke-width="1.5" stroke-linecap="round"/></g>""",  # noqa: E501
    "q": """<g id="black-queen" class="black queen" fill="#000" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g fill="#000" stroke="none"><circle cx="6" cy="12" r="2.75"/><circle cx="14" cy="9" r="2.75"/><circle cx="22.5" cy="8" r="2.75"/><circle cx="31" cy="9" r="2.75"/><circle cx="39" cy="12" r="2.75"/></g><path d="M9 26c8.5-1.5 21-1.5 27 0l2.5-12.5L31 25l-.3-14.1-5.2 13.6-3-14.5-3 14.5-5.2-13.6L14 25 6.5 13.5 9 26zM9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z" stroke-linecap="butt"/><path d="M11 38.5a35 35 1 0 0 23 0" fill="none" stroke-linecap="butt"/><path d="M11 29a35 35 1 0 1 23 0M12.5 31.5h20M11.5 34.5a35 35 1 0 0 22 0M10.5 37.5a35 35 1 0 0 24 0" fill="none" stroke="#fff"/></g>""",  # noqa: E501
    "r": """<g id="black-rook" class="black rook" fill="#000" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zM12.5 32l1.5-2.5h17l1.5 2.5h-20zM12 36v-4h21v4H12z" stroke-linecap="butt"/><path d="M14 29.5v-13h17v13H14z" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M14 16.5L11 14h23l-3 2.5H14zM11 14V9h4v2h5V9h5v2h5V9h4v5H11z" stroke-linecap="butt"/><path d="M12 35.5h21M13 31.5h19M14 29.5h17M14 16.5h17M11 14h23" fill="none" stroke="#fff" stroke-width="1" stroke-linejoin="miter"/></g>""",  # noqa: E501
    "B": """<g id="white-bishop" class="white bishop" fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><g fill="#fff" stroke-linecap="butt"><path d="M9 36c3.39-.97 10.11.43 13.5-2 3.39 2.43 10.11 1.03 13.5 2 0 0 1.65.54 3 2-.68.97-1.65.99-3 .5-3.39-.97-10.11.46-13.5-1-3.39 1.46-10.11.03-13.5 1-1.354.49-2.323.47-3-.5 1.354-1.94 3-2 3-2zM15 32c2.5 2.5 12.5 2.5 15 0 .5-1.5 0-2 0-2 0-2.5-2.5-4-2.5-4 5.5-1.5 6-11.5-5-15.5-11 4-10.5 14-5 15.5 0 0-2.5 1.5-2.5 4 0 0-.5.5 0 2zM25 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 1 1 5 0z"/></g><path d="M17.5 26h10M15 30h15m-7.5-14.5v5M20 18h5" stroke-linejoin="miter"/></g>""",  # noqa: E501
    "K": """<g id="white-king" class="white king" fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22.5 11.63V6M20 8h5" stroke-linejoin="miter"/><path d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5" fill="#fff" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M11.5 37c5.5 3.5 15.5 3.5 21 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-3.5-7.5-13-10.5-16-4-3 6 5 10 5 10V37z" fill="#fff"/><path d="M11.5 30c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0m-21 3.5c5.5-3 15.5-3 21 0"/></g>""",  # noqa: E501
    "N": """<g id="white-knight" class="white knight" fill="none" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18" style="fill:#ffffff; stroke:#000000;"/><path d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10" style="fill:#ffffff; stroke:#000000;"/><path d="M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z" style="fill:#000000; stroke:#000000;"/><path d="M 15 15.5 A 0.5 1.5 0 1 1 14,15.5 A 0.5 1.5 0 1 1 15 15.5 z" transform="matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)" style="fill:#000000; stroke:#000000;"/></g>""",  # noqa: E501
    "P": """<g id="white-pawn" class="white pawn"><path d="M22.5 9c-2.21 0-4 1.79-4 4 0 .89.29 1.71.78 2.38C17.33 16.5 16 18.59 16 21c0 2.03.94 3.84 2.41 5.03-3 1.06-7.41 5.55-7.41 13.47h23c0-7.92-4.41-12.41-7.41-13.47 1.47-1.19 2.41-3 2.41-5.03 0-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38 0-2.21-1.79-4-4-4z" fill="#fff" stroke="#000" stroke-width="1.5" stroke-linecap="round"/></g>""",  # noqa: E501
    "Q": """<g id="white-queen" class="white queen" fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM24.5 7.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM41 12a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM16 8.5a2 2 0 1 1-4 0 2 2 0 1 1 4 0zM33 9a2 2 0 1 1-4 0 2 2 0 1 1 4 0z"/><path d="M9 26c8.5-1.5 21-1.5 27 0l2-12-7 11V11l-5.5 13.5-3-15-3 15-5.5-14V25L7 14l2 12zM9 26c0 2 1.5 2 2.5 4 1 1.5 1 1 .5 3.5-1.5 1-1.5 2.5-1.5 2.5-1.5 1.5.5 2.5.5 2.5 6.5 1 16.5 1 23 0 0 0 1.5-1 0-2.5 0 0 .5-1.5-1-2.5-.5-2.5-.5-2 .5-3.5 1-2 2.5-2 2.5-4-8.5-1.5-18.5-1.5-27 0z" stroke-linecap="butt"/><path d="M11.5 30c3.5-1 18.5-1 22 0M12 33.5c6-1 15-1 21 0" fill="none"/></g>""",  # noqa: E501
    "R": """<g id="white-rook" class="white rook" fill="#fff" fill-rule="evenodd" stroke="#000" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M9 39h27v-3H9v3zM12 36v-4h21v4H12zM11 14V9h4v2h5V9h5v2h5V9h4v5" stroke-linecap="butt"/><path d="M34 14l-3 3H14l-3-3"/><path d="M31 17v12.5H14V17" stroke-linecap="butt" stroke-linejoin="miter"/><path d="M31 29.5l1.5 2.5h-20l1.5-2.5"/><path d="M11 14h23" fill="none" stroke-linejoin="miter"/></g>""",  # noqa: E501
}

# Noto Sans Bold glyph outlines (1000 units/em), matching Lichess. These paths
# keep standalone SVG/raster output independent of installed fonts. See LICENSE-NOTO.
COORDS = {
    '1': (572, 'M413 0H262V413Q262 430 262.5 455Q263 480 264 507Q265 534 266 555Q261 549 244.5 533.5Q228 518 214 506L132 440L59 531L289 714H413Z'),  # noqa: E501
    '2': (572, 'M539 0H40V105L219 286Q273 342 306 379.5Q339 417 354 447.5Q369 478 369 513Q369 556 345.5 577Q322 598 282 598Q241 598 202 579Q163 560 120 525L38 622Q69 649 103.5 672Q138 695 183.5 709.5Q229 724 293 724Q363 724 413.5 698.5Q464 673 491.5 629.5Q519 586 519 531Q519 472 495.5 423Q472 374 427.5 326Q383 278 320 220L228 134V127H539Z'),  # noqa: E501
    '3': (572, 'M511 554Q511 505 490.5 469Q470 433 435.5 410Q401 387 357 376V373Q443 363 487.5 321Q532 279 532 208Q532 146 501.5 96.5Q471 47 407.5 18.5Q344 -10 244 -10Q185 -10 134 0Q83 10 38 29V157Q84 134 134.5 122Q185 110 228 110Q309 110 341.5 138Q374 166 374 217Q374 247 359 267.5Q344 288 306.5 298.5Q269 309 202 309H148V425H203Q269 425 303.5 437.5Q338 450 350.5 471.5Q363 493 363 521Q363 559 339.5 580.5Q316 602 261 602Q227 602 199 593.5Q171 585 148.5 573Q126 561 109 550L39 654Q81 684 137.5 704Q194 724 272 724Q382 724 446.5 679.5Q511 635 511 554Z'),  # noqa: E501
    '4': (572, 'M555 148H469V0H322V148H17V253L330 714H469V265H555ZM322 386Q322 403 322.5 426.5Q323 450 324 473.5Q325 497 326 515.5Q327 534 328 541H324Q315 521 305 502Q295 483 281 463L150 265H322Z'),  # noqa: E501
    '5': (572, 'M300 456Q365 456 416 431Q467 406 496.5 358Q526 310 526 239Q526 162 494 106Q462 50 398.5 20Q335 -10 241 -10Q185 -10 135.5 0Q86 10 49 29V159Q86 140 138 126.5Q190 113 236 113Q281 113 311.5 125Q342 137 358 162Q374 187 374 226Q374 279 339 306.5Q304 334 231 334Q203 334 173 328.5Q143 323 123 318L63 350L90 714H477V586H222L209 446Q226 449 245.5 452.5Q265 456 300 456Z'),  # noqa: E501
    '6': (572, 'M35 303Q35 365 44.0763 425Q53.1525 485 75.5763 538.5Q98 592 138.728 633.5Q179.455 675 241.647 698.5Q303.84 722 393 722Q414 722 442 720.5Q470 719 489 715V594Q470 599 447.5 601.5Q425 604 402.75 604Q336 604 292.5 588Q249 572 224 542Q199 512 187.5 471.5Q176 431 174 381H179.714Q194 405 214.5 423.5Q235 442 265.093 453Q295.185 464 335 464Q398 464 443.5 437.5Q489 411 514 360.479Q539 309.958 539 238.083Q539 161 509.5 105Q480 49 425.59 19.5Q371.181 -10 295.761 -10Q240.813 -10 193.406 9Q146 28 110.5 66.5Q75 105 55 164.201Q35 223.402 35 303ZM292.868 111Q337 111 365 141.5Q393 172 393 236.206Q393 287.571 368.757 317.785Q344.514 348 296.027 348Q263 348 237.954 333.256Q212.908 318.512 198.954 295.829Q185 273.146 185 248.951Q185 224 192 199.5Q199 175 212.66 154.907Q226.32 134.814 246.284 122.907Q266.249 111 292.868 111Z'),  # noqa: E501
    '7': (572, 'M111 0 379 587H27V714H539V619L269 0Z'),  # noqa: E501
    '8': (572, 'M286 723Q348 723 399.5 704Q451 685 482.5 647Q514 609 514 551Q514 508 497 475.5Q480 443 451.5 419Q423 395 386 377Q424 357 458.5 330.5Q493 304 514.5 268.5Q536 233 536 185Q536 126 504.5 82Q473 38 416.5 14Q360 -10 286 -10Q206 -10 150 13Q94 36 64.5 79Q35 122 35 181Q35 230 53.5 266Q72 302 103 328.5Q134 355 172 373Q140 393 114 418.5Q88 444 72.5 476.5Q57 509 57 552Q57 609 89 647Q121 685 173.5 704Q226 723 286 723ZM175 190Q175 151 202.5 126Q230 101 284 101Q340 101 368 125Q396 149 396 189Q396 216 380 236.5Q364 257 340.5 273.5Q317 290 292 304L279 311Q248 297 224.5 279Q201 261 188 239.5Q175 218 175 190ZM285 613Q248 613 223.5 594Q199 575 199 540Q199 516 211 497.5Q223 479 243 465.5Q263 452 286 440Q309 451 328.5 464Q348 477 360 495.5Q372 514 372 540Q372 575 347.5 594Q323 613 285 613Z'),  # noqa: E501
    'a': (599, 'M317.279 555.98Q419.549 555.98 473.869 507.49Q528.189 459 528.189 363.78V0H422.71L393.51 74.17H389.51Q366.51 45.17 342.01 26.28Q317.51 7.39001 285.705 -1.30499Q253.9 -10 208.29 -10Q160.29 -10 122.095 8.52496Q83.9001 27.0499 61.9001 64.9899Q39.9001 102.93 39.9001 161.2Q39.9001 246.69 101.985 289.885Q164.07 333.08 286.53 337.69L378.14 340.69V358.05Q378.14 406.75 355.85 426.346Q333.56 445.941 293.98 445.941Q257.47 445.941 218.25 434.051Q179.03 422.161 141.15 404.941L97.0702 507.54Q140 530.2 195.625 543.09Q251.25 555.98 317.279 555.98ZM320.71 250.46Q249.989 247.68 221.859 226.695Q193.729 205.71 193.729 167.52Q193.729 132.889 213.604 116.464Q233.479 100.039 265.499 100.039Q313.37 100.039 345.865 128.084Q378.36 156.13 378.36 207.9V252.85Z'),  # noqa: E501
    'b': (632, 'M224.239 582.93Q224.239 551.98 222.434 524.005Q220.629 496.03 218.019 475.37H224.239Q246.019 509.37 282.969 532.675Q319.919 555.98 378.649 555.98Q470.109 555.98 527.254 484.345Q584.399 412.71 584.399 274.1Q584.399 180.88 558.119 117.355Q531.839 53.83 484.584 21.915Q437.329 -10 374.989 -10Q315.039 -10 280.529 11.475Q246.019 32.9501 224.239 59.4602H214.189L188.869 0H73.4103V760H224.239ZM330.1 436.011Q292.009 436.011 268.464 420.001Q244.919 403.991 234.579 371.385Q224.239 338.78 224.239 287.47V269.44Q224.239 191.789 247.589 152.099Q270.939 112.409 332.1 112.409Q380.92 112.409 405.746 154.124Q430.571 195.839 430.571 275.71Q430.571 355.36 406.051 395.686Q381.53 436.011 330.1 436.011Z'),  # noqa: E501
    'c': (516, 'M310.98 -10Q232.05 -10 172.585 19.61Q113.12 49.22 80.0952 111.44Q47.0702 173.66 47.0702 270.49Q47.0702 370.71 83.1201 433.625Q119.17 496.54 181.16 526.455Q243.15 556.37 322.47 556.37Q369.569 556.37 412.144 546.345Q454.719 536.32 487.869 519.44L443.179 404.961Q413.08 417.621 382.605 426.231Q352.13 434.841 321.69 434.841Q282.939 434.841 255.869 416.721Q228.799 398.601 214.849 362.445Q200.899 326.29 200.899 271.49Q200.899 217.25 215.154 182.009Q229.409 146.769 256.089 129.564Q282.769 112.359 320.349 112.359Q362.52 112.359 400.899 123.799Q439.279 135.239 472.769 154.339V31.1997Q441.499 12.3198 402.594 1.15991Q363.69 -10 310.98 -10Z'),  # noqa: E501
    'd': (632, 'M252.431 -10Q161.58 -10 104.325 61.6349Q47.0702 133.27 47.0702 272.49Q47.0702 412.93 105.13 484.65Q163.19 556.37 257.09 556.37Q296.041 556.37 324.601 545.87Q353.161 535.37 374.331 517.065Q395.501 498.76 410.281 476.15H415.061Q412.231 494.03 409.731 526.544Q407.231 559.059 407.231 586.259V760H558.669V0H443.14L413.061 70.78H407.231Q393.061 48.39 371.891 30.195Q350.721 12 321.356 1Q291.991 -10 252.431 -10ZM305.47 110.579Q366.68 110.579 391.725 146.404Q416.771 182.229 417.381 255.49V270.88Q417.381 349.53 393.225 391.746Q369.07 433.961 303.86 433.961Q255.429 433.961 227.859 391.796Q200.289 349.63 200.289 269.88Q200.289 190.349 227.859 150.464Q255.429 110.579 305.47 110.579Z'),  # noqa: E501
    'e': (597, 'M306.03 556.37Q382.349 556.37 437.339 527.59Q492.329 498.81 521.879 443.395Q551.429 387.98 551.429 307.66V235.14H200.289Q202.289 173.42 238.249 137.814Q274.209 102.209 340.789 102.209Q392.399 102.209 434.034 112.014Q475.669 121.819 520.109 142.259V28.6599Q479.719 8.82996 435.144 -0.585022Q390.569 -10 325.2 -10Q244.32 -10 181.55 20.085Q118.78 50.17 82.9251 112.585Q47.0702 175 47.0702 269.49Q47.0702 364.81 79.6201 428.42Q112.17 492.03 170.55 524.2Q228.93 556.37 306.03 556.37ZM309.91 448.991Q264.599 448.991 236.639 420.385Q208.679 391.78 203.509 335.64H410.02Q410.02 368.83 398.765 393.985Q387.51 419.14 365.415 434.065Q343.32 448.991 309.91 448.991Z'),  # noqa: E501
    'f': (387, 'M380 434H251V0H102V434H20V506L102 546V586Q102 656 125.5 694.5Q149 733 192.5 749Q236 765 295 765Q339 765 374.5 758Q410 751 432 742L394 633Q377 638 357 642.5Q337 647 311 647Q280 647 265.5 628Q251 609 251 580V546H380Z'),  # noqa: E501
    'g': (632, 'M255.48 556.37Q311.53 556.37 349.76 534.845Q387.991 513.32 413.111 477.2H417.501L430.721 546.37H558.669V-4.12012Q558.669 -81.3401 529.009 -133.755Q499.349 -186.17 438.725 -213.085Q378.1 -240 285.12 -240Q221.63 -240 171.605 -232.415Q121.58 -224.83 78.0905 -207.22V-78.6411Q123.41 -99.2511 170.75 -109.946Q218.09 -120.641 280.36 -120.641Q342.651 -120.641 375.441 -92.0358Q408.231 -63.4305 408.231 -11.7301V2.56006Q408.231 15.17 409.536 34.9251Q410.841 54.6802 413.061 70.39H407.841Q385.331 34.2699 347.796 12.1349Q310.26 -10 254.041 -10Q159.8 -10 103.435 63Q47.0702 136 47.0702 272.49Q47.0702 407.81 104.045 482.09Q161.02 556.37 255.48 556.37ZM304.08 437.231Q270.669 437.231 247.404 418.111Q224.139 398.991 212.519 361.835Q200.899 324.68 200.899 270.27Q200.899 187.74 227.334 148.049Q253.769 108.359 306.69 108.359Q336.71 108.359 357.78 117.029Q378.85 125.699 392.055 143.319Q405.26 160.939 411.516 188.009Q417.771 215.08 417.771 252.1V274.32Q417.771 329.02 406.76 365.175Q395.75 401.331 370.815 419.281Q345.88 437.231 304.08 437.231Z'),  # noqa: E501
    'h': (650, 'M224.239 607.148Q224.239 562.529 221.824 529.109Q219.409 495.69 217.409 476.03H225.239Q242.849 504.64 266.714 522.03Q290.579 539.42 320.139 547.7Q349.699 555.98 383.089 555.98Q441.989 555.98 485.794 535.175Q529.599 514.37 554.344 470.455Q579.089 426.54 579.089 355.71V0H428.041V317.991Q428.041 376.891 406.97 406.451Q385.9 436.011 340.98 436.011Q296.839 436.011 271.379 415.036Q245.919 394.061 235.079 354.221Q224.239 314.381 224.239 256.26V0H73.4103V760H224.239Z'),  # noqa: E501
}

XX = """<g id="xx"><path d="M35.865 9.135a1.89 1.89 0 0 1 0 2.673L25.173 22.5l10.692 10.692a1.89 1.89 0 0 1 0 2.673 1.89 1.89 0 0 1-2.673 0L22.5 25.173 11.808 35.865a1.89 1.89 0 0 1-2.673 0 1.89 1.89 0 0 1 0-2.673L19.827 22.5 9.135 11.808a1.89 1.89 0 0 1 0-2.673 1.89 1.89 0 0 1 2.673 0L22.5 19.827 33.192 9.135a1.89 1.89 0 0 1 2.673 0z" fill="#000" stroke="#fff" stroke-width="1.688"/></g>"""  # noqa: E501

CHECK_GRADIENT = """<radialGradient id="check_gradient" r="0.5"><stop offset="0%" stop-color="#ff0000" stop-opacity="1.0" /><stop offset="50%" stop-color="#e70000" stop-opacity="1.0" /><stop offset="100%" stop-color="#9e0000" stop-opacity="0.0" /></radialGradient>"""  # noqa: E501

DEFAULT_COLORS = {
    "square light": "#ffce9e",
    "square dark": "#d18b47",
    "square dark lastmove": "#aaa23b",
    "square light lastmove": "#cdd16a",
    "margin": "#212121",
    "inner border": "#111",
    "outer border": "#111",
    "coord": "#e5e5e5",
    "arrow green": "#15781B80",
    "arrow red": "#88202080",
    "arrow yellow": "#e68f00b3",
    "arrow blue": "#00308880",
}

CHESS_COM_ARROW_COLORS = {
    # Chess.com applies both an alpha of 0.8 to the fill and an opacity of 0.8
    # to the polygon. These alpha values preserve the resulting appearance.
    "arrow green": "#9fcf3fa3",
    "arrow red": "#f8553fa3",
    "arrow yellow": "#ffaa00a3",
    "arrow blue": "#48c1f9a3",
}

LICHESS_ARROW_COLORS = {
    "arrow green": "#15781B",
    "arrow red": "#882020",
    "arrow yellow": "#e68f00",
    "arrow blue": "#003088",
}

_CSS_NAMED_COLORS = frozenset("""
    aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet brown
    burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan darkgoldenrod
    darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen darkslateblue
    darkslategray darkslategrey darkturquoise darkviolet deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen
    fuchsia gainsboro ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred
    indigo ivory khaki lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow lightgray
    lightgreen lightgrey lightpink lightsalmon lightseagreen lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen
    linen magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue mediumspringgreen mediumturquoise mediumvioletred
    midnightblue mintcream mistyrose moccasin navajowhite navy oldlace olive olivedrab orange orangered orchid
    palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum powderblue purple rebeccapurple
    red rosybrown royalblue saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue
    slategray slategrey snow springgreen steelblue tan teal thistle tomato turquoise violet wheat
    white whitesmoke yellow yellowgreen
""".split())


class Arrow:
    """Details of an arrow to be drawn."""

    tail: Square
    """Start square of the arrow."""

    head: Square
    """End square of the arrow."""

    color: str
    """Arrow color."""

    def __init__(self, tail: Square, head: Square, *, color: str = "green") -> None:
        self.tail = tail
        self.head = head
        self.color = color

    def pgn(self) -> str:
        """
        Returns the arrow in the format used by ``[%csl ...]`` and
        ``[%cal ...]`` PGN annotations, e.g., ``Ga1`` or ``Ya2h2``.

        Colors other than ``red``, ``yellow``, and ``blue`` default to green.
        """
        if self.color == "red":
            color = "R"
        elif self.color == "yellow":
            color = "Y"
        elif self.color == "blue":
            color = "B"
        else:
            color = "G"

        if self.tail == self.head:
            return f"{color}{chess.SQUARE_NAMES[self.tail]}"
        else:
            return f"{color}{chess.SQUARE_NAMES[self.tail]}{chess.SQUARE_NAMES[self.head]}"

    def __str__(self) -> str:
        return self.pgn()

    def __repr__(self) -> str:
        return f"Arrow({chess.SQUARE_NAMES[self.tail].upper()}, {chess.SQUARE_NAMES[self.head].upper()}, color={self.color!r})"

    @classmethod
    def from_pgn(cls, pgn: str) -> Arrow:
        """
        Parses an arrow from the format used by ``[%csl ...]`` and
        ``[%cal ...]`` PGN annotations, e.g., ``Ga1`` or ``Ya2h2``.

        Also allows skipping the color prefix, defaulting to green.

        :raises: :exc:`ValueError` if the format is invalid.
        """
        if pgn.startswith("G"):
            color = "green"
            pgn = pgn[1:]
        elif pgn.startswith("R"):
            color = "red"
            pgn = pgn[1:]
        elif pgn.startswith("Y"):
            color = "yellow"
            pgn = pgn[1:]
        elif pgn.startswith("B"):
            color = "blue"
            pgn = pgn[1:]
        else:
            color = "green"

        tail = chess.parse_square(pgn[:2])
        head = chess.parse_square(pgn[2:]) if len(pgn) > 2 else tail
        return cls(tail, head, color=color)


class SvgWrapper(str):
    def _repr_svg_(self) -> SvgWrapper:
        return self

    def _repr_html_(self) -> SvgWrapper:
        return self


def _svg(viewbox: int, size: Optional[int]) -> ET.Element:
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "xmlns:xlink": "http://www.w3.org/1999/xlink",
        "viewBox": f"0 0 {viewbox:d} {viewbox:d}",
    })

    if size is not None:
        svg.set("width", str(size))
        svg.set("height", str(size))

    return svg


def _attrs(attrs: Dict[str, Union[str, int, float, None]]) -> Dict[str, str]:
    return {k: str(v) for k, v in attrs.items() if v is not None}


def _select_color(colors: Dict[str, str], color: str) -> Tuple[str, float]:
    return _color(colors.get(color, DEFAULT_COLORS[color]))


def _color(color: str) -> Tuple[str, float]:
    if color.startswith("#"):
        try:
            if len(color) == 5:
                return color[:4], int(color[4], 16) / 0xf
            elif len(color) == 9:
                return color[:7], int(color[7:], 16) / 0xff
        except ValueError:
            pass  # Ignore invalid hex value
    return color, 1.0


def _is_valid_color(color: str, opacity: float, *, require_opaque: bool) -> bool:
    normalized = color.strip().lower()
    valid_opacity = opacity == 1.0 if require_opaque else opacity > 0.0
    return valid_opacity and (
        normalized in _CSS_NAMED_COLORS
        or (
            len(normalized) in (4, 7)
            and normalized.startswith("#")
            and all(character in "0123456789abcdef" for character in normalized[1:])
        )
    )


def _square_origin(square: Square, *, orientation: Color, board_offset: int) -> Tuple[float, float]:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    return (
        board_offset + (file_index if orientation else 7 - file_index) * SQUARE_SIZE,
        board_offset + (7 - rank_index if orientation else rank_index) * SQUARE_SIZE,
    )


def _square_center(square: Square, *, orientation: Color, board_offset: int) -> Tuple[float, float]:
    x, y = _square_origin(square, orientation=orientation, board_offset=board_offset)
    return x + SQUARE_SIZE / 2, y + SQUARE_SIZE / 2


def _box_from_center(cx: float, cy: float, radius: float) -> Tuple[float, float, float, float]:
    return cx - radius, cy - radius, cx + radius, cy + radius


def _normalize_legal_moves(
    board: Optional[chess.BaseBoard],
    legal_moves: Iterable[chess.Move],
    *,
    legal_move_style: LegalMoveStyle,
) -> Tuple[DestinationMarker, ...]:
    requested = tuple(legal_moves)
    if not requested:
        return ()
    if not isinstance(board, chess.Board):
        raise ValueError("legal_moves require a chess.Board")
    source_squares = {move.from_square for move in requested}
    if len(source_squares) != 1:
        raise ValueError("legal_moves must share one source square")
    seen_destinations: set[Square] = set()
    normalized: list[DestinationMarker] = []
    for move in requested:
        if move not in board.legal_moves:
            raise ValueError(f"legal_moves contains an illegal move: {move.uci()}")
        destinations = (move,)
        if legal_move_style == "lichess" and board.is_castling(move):
            # Chessground exposes both accepted castling gestures.
            king_square = chess.square(
                6 if board.is_kingside_castling(move) else 2,
                chess.square_rank(move.from_square),
            )
            rook_destination = board._to_chess960(move)
            destinations = (
                (chess.Move(move.from_square, king_square), rook_destination)
                if king_square != move.from_square
                else (rook_destination,)
            )
        for destination in destinations:
            if destination.to_square in seen_destinations:
                continue
            seen_destinations.add(destination.to_square)
            normalized.append(
                DestinationMarker(
                    destination.to_square,
                    "dot" if board.piece_at(destination.to_square) is None else "capture",
                )
            )
    return tuple(normalized)


def _oriented_box(
    points: Iterable[Tuple[float, float]],
    direction: Tuple[float, float],
) -> Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
]:
    """Tight enclosing rectangle aligned with *direction*."""
    points = tuple(points)
    if len(points) < 3:
        raise ValueError("oriented bounds require at least three points")
    length = math.hypot(*direction)
    if not length:
        raise ValueError("oriented bounds require a direction")
    ux, uy = direction[0] / length, direction[1] / length
    vx, vy = -uy, ux
    along = [x * ux + y * uy for x, y in points]
    across = [x * vx + y * vy for x, y in points]
    lo_u, hi_u = min(along), max(along)
    lo_v, hi_v = min(across), max(across)
    return (
        (lo_u * ux + lo_v * vx, lo_u * uy + lo_v * vy),
        (hi_u * ux + lo_v * vx, hi_u * uy + lo_v * vy),
        (hi_u * ux + hi_v * vx, hi_u * uy + hi_v * vy),
        (lo_u * ux + hi_v * vx, lo_u * uy + hi_v * vy),
    )


def _points_bbox(points: Iterable[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    points = tuple(points)
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _coordinates(*, orientation: Color, style: CoordinateStyle, colors: Dict[str, str],
                 offset: float, size: Optional[int], full_size: float) -> ET.Element:
    """In-board labels; Lichess desktop CSS or Chess.com's 100-unit SVG layout."""
    group = ET.Element("g", {"class": f"coordinates {style}", "pointer-events": "none"})
    files = list(chess.FILE_NAMES if orientation else reversed(chess.FILE_NAMES))
    ranks = list(reversed(chess.RANK_NAMES)) if orientation else list(chess.RANK_NAMES)
    edge = 8 * SQUARE_SIZE
    pixel = full_size / size if size else 1.0
    # Lichess coords.files uses flex: 1 1 auto, so glyph advances affect cell widths.
    advances = [COORDS[name][0] * 12 / 1000 * pixel for name in files]
    free = (edge - sum(advances)) / 8
    file_x = 4 * pixel
    for axis, names in (("rank", ranks), ("file", files)):
        for index, name in enumerate(names):
            # Screen top-left is always light. Use the opposite square color.
            light = (index % 2 == 0) if axis == "rank" and style == "chess.com" else index % 2 == 1
            key = "coord dark" if light else "coord light"
            color, opacity = _color(colors[key]) if key in colors else _select_color(colors, "square dark" if light else "square light")
            if style == "lichess":
                # Desktop Lichess: 12px bold Noto Sans, ranks top:1px/right:0,
                # width:.8em; files bottom:0, height:1.4em, padding-left:4px.
                x = edge - 9.6 * pixel if axis == "rank" else file_x
                y = index * SQUARE_SIZE + 14 * pixel if axis == "rank" else edge - 3.8 * pixel
                node = ET.SubElement(group, "path", _attrs({
                    "d": COORDS[name][1], "data-coordinate": name, "class": axis,
                    "transform": f"translate({offset + x:g},{offset + y:g}) scale({.012 * pixel:g},{-.012 * pixel:g})",
                    "fill": color, "opacity": opacity if opacity < 1 else None,
                }))
                if axis == "file":
                    file_x += free + advances[index]
            else:
                # Exact SVG positions observed on Chess.com analysis (2026.9.5).
                x = .75 if axis == "rank" else 10 + index * 12.5
                y = (3.5 if index == 0 else 3.25 + index * 12.5) if axis == "rank" else 99
                node = ET.SubElement(group, "text", _attrs({
                    "x": offset + x * edge / 100, "y": offset + y * edge / 100,
                    "font-size": 2.8 * edge / 100, "font-weight": 600,
                    "font-family": "-apple-system, system-ui, Segoe UI, Helvetica, Arial, Liberation Sans, sans-serif",
                    "fill": color, "opacity": opacity if opacity < 1 else None, "class": axis,
                }))
                node.text = name
    return group



def _piece_code(piece: chess.Piece, *, piece_set: str) -> str:
    """Returns a filename stem like ``wP`` or ``bN`` for the piece."""
    if piece_set == "mono":
        return piece.symbol().upper()
    return f"{'w' if piece.color else 'b'}{piece.symbol().upper()}"


def _embedded_piece(piece_svg: str) -> ET.Element:
    root = ET.fromstring(piece_svg)
    view_box = root.get("viewBox")
    if view_box is None:
        view_box = f"0 0 {root.get('width')} {root.get('height')}"

    try:
        min_x, min_y, width, height = map(float, view_box.replace(",", " ").split())
    except ValueError as error:
        raise ValueError("piece SVG must have a valid viewBox or numeric width and height") from error

    if root.get("viewBox") is None or min_x or min_y:
        if min_x or min_y:
            namespace, separator, _ = root.tag.rpartition("}")
            group_tag = f"{namespace}}}g" if separator else "g"
            group = ET.Element(group_tag, {"transform": f"translate({-min_x:g} {-min_y:g})"})
            group.extend(list(root))
            root[:] = [group]
        root.set("viewBox", f"0 0 {width:g} {height:g}")
        piece_svg_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    else:
        piece_svg_bytes = piece_svg.encode("utf-8")

    data = base64.b64encode(piece_svg_bytes).decode("ascii")
    href = f"data:image/svg+xml;base64,{data}"
    return ET.Element("image", {
        "width": str(SQUARE_SIZE),
        "height": str(SQUARE_SIZE),
        "href": href,
        "xlink:href": href,
    })


@lru_cache(maxsize=None)
def load_pieces(piece_set: str) -> dict[str, str]:
    """
    Loads piece SVGs from ``chess/piece/<piece_set>``, caching results in-process.
    """
    piece_set_dir = Path(__file__).resolve().parent / "piece" / piece_set
    if not piece_set_dir.is_dir():
        raise FileNotFoundError(f"Piece set not found: {piece_set!r}")

    pieces: dict[str, str] = {}
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            piece = chess.Piece(piece_type, color)
            piece_code = _piece_code(piece, piece_set=piece_set)
            if piece_code in pieces:
                # This should only happen for `piece_set == 'mono'`.
                continue

            pieces[piece_code] = (piece_set_dir / f"{piece_code}.svg").read_text(encoding="utf-8")

    return pieces


def piece(piece: chess.Piece, size: Optional[int] = None, *, piece_set: Optional[str] = None) -> str:
    """
    Renders the given :class:`chess.Piece` as an SVG image.

    >>> import chess
    >>> import chess.svg
    >>>
    >>> chess.svg.piece(chess.Piece.from_symbol("R"))  # doctest: +SKIP

    .. image:: ../docs/wR.svg
        :alt: R
    """
    svg = _svg(SQUARE_SIZE, size)
    if piece_set is None:
        svg.append(ET.fromstring(PIECES[piece.symbol()]))
    else:
        svg.append(_embedded_piece(load_pieces(piece_set)[_piece_code(piece, piece_set=piece_set)]))
    return SvgWrapper(ET.tostring(svg).decode("utf-8"))


def board(board: Optional[chess.BaseBoard] = None, *,
          orientation: Color = chess.WHITE,
          lastmove: Optional[chess.Move] = None,
          check: Optional[Square] = None,
          arrows: Iterable[Union[Arrow, Tuple[Square, Square]]] = [],
          arrow_style: ArrowStyle = "lichess",
          fill: Dict[Square, str] = {},
          squares: Optional[IntoSquareSet] = None,
          size: Optional[int] = None,
          css_size: Optional[float] = None,
          device_pixel_ratio: float = 1.0,
          coordinates: bool = True,
          coordinate_style: CoordinateStyle = "lichess",
          colors: Dict[str, str] = {},
          borders: bool = False,
          style: Optional[str] = None,
          piece_set: Optional[str] = None,
          legal_moves: Iterable[chess.Move] = (),
          destination_markers: Iterable[DestinationMarker] = (),
          legal_move_style: LegalMoveStyle = "lichess",
          user_highlights: Iterable[UserHighlight] = (),
          ghost_squares: Iterable[Square] = ()) -> "SvgWrapper":
    """
    Renders a board with pieces and/or selected squares as an SVG image.
    Use :func:`board_with_annotations` when semantic overlay bounds are needed.

    :param board: A :class:`chess.BaseBoard` for a chessboard with pieces, or
        ``None`` (the default) for a chessboard without pieces.
    :param orientation: The point of view, defaulting to ``chess.WHITE``.
    :param lastmove: A :class:`chess.Move` to be highlighted.
    :param check: A square to be marked indicating a check.
    :param arrows: A list of :class:`~chess.svg.Arrow` objects, like
        ``[chess.svg.Arrow(chess.E2, chess.E4)]``, or a list of tuples, like
        ``[(chess.E2, chess.E4)]``. An arrow from a square pointing to the same
        square is drawn as a circle, like ``[(chess.E2, chess.E2)]``.
    :param arrow_style: The arrow geometry and default palette. ``"lichess"``
        (the default) uses rounded shafts with triangular markers, matching
        Chessground. All Lichess arrows share the site's opacity layer, so
        custom arrow colors must be opaque. ``"chess.com"`` uses filled
        polygons and renders knight moves as L-shaped arrows.
    :param fill: A dictionary mapping squares to a colors that they should be
        filled with.
    :param squares: A :class:`chess.SquareSet` with selected squares to mark
        with an X.
    :param size: The size of the image in pixels (e.g., ``400`` for a 400 by
        400 board), or ``None`` (the default) for no size limit.
    :param device_pixel_ratio: CSS device-pixel scale (default 1), used to snap
        Chess.com capture borders like Chromium. Does not change output size.
    :param css_size: Logical board size in CSS pixels for size-dependent capture
        rings. Defaults to *size*, or the SVG viewBox size when *size* is absent.
        Set separately when resizing a screenshot preview; this does not change
        image dimensions or annotation coordinates.
    :param coordinates: Render in-board file/rank labels; never adds a gutter.
    :param coordinate_style: ``"lichess"`` (default) or ``"chess.com"``.
        Lichess uses desktop 12px Noto Sans Bold outlines. Chess.com uses
        its native system-font stack, so glyphs depend on the viewing platform.
    :param colors: A dictionary to override default colors. Possible keys are
        ``square light``, ``square dark``, ``square light lastmove``,
        ``square dark lastmove``, ``coord light``, ``coord dark``,
        ``outer border``, ``arrow green``, ``arrow blue``, ``arrow red``,
        and ``arrow yellow``. Values should look like ``#ffce9e`` (opaque),
        or ``#15781B80`` (transparent).
    :param borders: Pass ``True`` to enable a border around the board edge
        (coordinates do not add a margin).
    :param style: A CSS stylesheet to include in the SVG image.
    :param legal_moves: Legal moves from one source square whose destinations
        should be marked. Promotion variants sharing a destination are
        deduplicated. Lichess-style castling shows both the king destination
        and the rook square.
    :param legal_move_style: The legal-destination geometry, either
        ``"lichess"`` (the default) or ``"chess.com"``, for both *legal_moves*
        and *destination_markers*.
    :param destination_markers: Explicit :class:`DestinationMarker` values, without
        requiring a legal position, source square, or matching square occupancy.
        Squares must be distinct. Cannot be combined with nonempty *legal_moves*.
    :param user_highlights: Foreground square-circle annotations with canonical
        colors and a Lichess or Chess.com palette.
    :param ghost_squares: Occupied squares whose pieces are shown as Lichess
        drag ghosts, at 0.3 opacity above board overlays. This changes only
        their appearance, not the board or overlay annotations.

    >>> import chess
    >>> import chess.svg
    >>>
    >>> board = chess.Board("8/8/8/8/4N3/8/8/8 w - - 0 1")
    >>>
    >>> chess.svg.board(
    ...     board,
    ...     fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
    ...     arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cc")],
    ...     squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
    ...     size=350,
    ... )  # doctest: +SKIP

    .. image:: ../docs/Ne4.svg
        :alt: 8/8/8/8/4N3/8/8/8
    """
    return _render_board(
        board,
        orientation=orientation,
        lastmove=lastmove,
        check=check,
        arrows=arrows,
        arrow_style=arrow_style,
        fill=fill,
        squares=squares,
        size=size,
        css_size=css_size,
        device_pixel_ratio=device_pixel_ratio,
        coordinates=coordinates,
        coordinate_style=coordinate_style,
        colors=colors,
        borders=borders,
        style=style,
        piece_set=piece_set,
        legal_moves=legal_moves,
        destination_markers=destination_markers,
        legal_move_style=legal_move_style,
        user_highlights=user_highlights,
        ghost_squares=ghost_squares,
    ).svg


def board_with_annotations(board: Optional[chess.BaseBoard] = None, *,
                           orientation: Color = chess.WHITE,
                           lastmove: Optional[chess.Move] = None,
                           check: Optional[Square] = None,
                           arrows: Iterable[Union[Arrow, Tuple[Square, Square]]] = [],
                           arrow_style: ArrowStyle = "lichess",
                           fill: Dict[Square, str] = {},
                           squares: Optional[IntoSquareSet] = None,
                           size: Optional[int] = None,
                           css_size: Optional[float] = None,
                           device_pixel_ratio: float = 1.0,
                           coordinates: bool = True,
                           coordinate_style: CoordinateStyle = "lichess",
                           colors: Dict[str, str] = {},
                           borders: bool = False,
                           piece_set: Optional[str] = None,
                           legal_moves: Iterable[chess.Move] = (),
                           destination_markers: Iterable[DestinationMarker] = (),
                           legal_move_style: LegalMoveStyle = "lichess",
                           user_highlights: Iterable[UserHighlight] = (),
                           ghost_squares: Iterable[Square] = ()) -> BoardRenderResult:
    """Renders a board SVG with renderer-owned semantic overlay geometry.

    Parameters match :func:`board`, except arbitrary CSS ``style`` is
    deliberately unavailable because it could invalidate the returned bounds.
    """
    return _render_board(
        board,
        orientation=orientation,
        lastmove=lastmove,
        check=check,
        arrows=arrows,
        arrow_style=arrow_style,
        fill=fill,
        squares=squares,
        size=size,
        css_size=css_size,
        device_pixel_ratio=device_pixel_ratio,
        coordinates=coordinates,
        coordinate_style=coordinate_style,
        colors=colors,
        borders=borders,
        piece_set=piece_set,
        legal_moves=legal_moves,
        destination_markers=destination_markers,
        legal_move_style=legal_move_style,
        user_highlights=user_highlights,
        ghost_squares=ghost_squares,
    )


def _render_board(board: Optional[chess.BaseBoard] = None, *,
                  orientation: Color = chess.WHITE,
                  lastmove: Optional[chess.Move] = None,
                  check: Optional[Square] = None,
                  arrows: Iterable[Union[Arrow, Tuple[Square, Square]]] = [],
                  arrow_style: ArrowStyle = "lichess",
                  fill: Dict[Square, str] = {},
                  squares: Optional[IntoSquareSet] = None,
                  size: Optional[int] = None,
                  css_size: Optional[float] = None,
                  device_pixel_ratio: float = 1.0,
                  coordinates: bool = True,
                  coordinate_style: CoordinateStyle = "lichess",
                  colors: Dict[str, str] = {},
                  borders: bool = False,
                  style: Optional[str] = None,
                  piece_set: Optional[str] = None,
                  legal_moves: Iterable[chess.Move] = (),
                  destination_markers: Iterable[DestinationMarker] = (),
                  legal_move_style: LegalMoveStyle = "lichess",
                  user_highlights: Iterable[UserHighlight] = (),
                  ghost_squares: Iterable[Square] = ()) -> BoardRenderResult:
    """Builds the shared SVG and annotation result for the public renderers."""
    if not math.isfinite(device_pixel_ratio) or not 0 < device_pixel_ratio <= 16:
        raise ValueError("device_pixel_ratio must be finite and in (0, 16]")
    if css_size is not None and (not math.isfinite(css_size) or css_size <= 0):
        raise ValueError("css_size must be finite and positive")
    if coordinate_style not in ["lichess", "chess.com"]:
        raise ValueError(f"unsupported coordinate style: {coordinate_style!r}")
    if arrow_style not in ["lichess", "chess.com"]:
        raise ValueError(f"unsupported arrow style: {arrow_style!r}")
    if legal_move_style not in ["lichess", "chess.com"]:
        raise ValueError(f"unsupported legal move style: {legal_move_style!r}")
    ghost_squares = frozenset(ghost_squares)
    if any(
        type(square) is not int or square not in chess.SQUARES
        or board is None or board.piece_at(square) is None
        for square in ghost_squares
    ):
        raise ValueError("ghost_squares must contain occupied board squares")
    legal_moves = tuple(legal_moves)
    markers = tuple(destination_markers)
    if markers and legal_moves:
        raise ValueError("destination_markers and legal_moves cannot be combined")
    if any(not isinstance(marker, DestinationMarker) for marker in markers):
        raise TypeError("destination_markers must contain DestinationMarker values")
    if len({marker.square for marker in markers}) != len(markers):
        raise ValueError("destination_markers must have distinct squares")
    if not markers:
        markers = _normalize_legal_moves(board, legal_moves, legal_move_style=legal_move_style)
    arrows = tuple(arrows)
    highlights = tuple(user_highlights)
    if any(not isinstance(highlight, UserHighlight) for highlight in highlights):
        raise TypeError("user_highlights must contain UserHighlight values")
    highlight_palettes: Dict[Square, LegalMoveStyle] = {}
    for highlight in highlights:
        if (
            highlight.square in highlight_palettes
            and highlight_palettes[highlight.square] != highlight.palette
        ):
            raise ValueError("user highlights on one square must use one palette")
        highlight_palettes[highlight.square] = highlight.palette
    annotations: list[OverlayAnnotation] = []

    outer_border = 1 if borders else 0
    board_offset = outer_border
    full_size = 2 * outer_border + 8 * SQUARE_SIZE
    svg = _svg(full_size, size)

    if style:
        ET.SubElement(svg, "style").text = style

    if board:
        desc = ET.SubElement(svg, "desc")
        asciiboard = ET.SubElement(desc, "pre")
        asciiboard.text = str(board)

    defs = ET.SubElement(svg, "defs")
    if board:
        if piece_set is None:
            for piece_color in chess.COLORS:
                for piece_type in chess.PIECE_TYPES:
                    if board.pieces_mask(piece_type, piece_color):
                        defs.append(ET.fromstring(PIECES[chess.Piece(piece_type, piece_color).symbol()]))
        else:
            pieces = load_pieces(piece_set)
            piece_codes = sorted({_piece_code(piece, piece_set=piece_set) for piece in board.piece_map().values()})
            for piece_code in piece_codes:
                piece_image = _embedded_piece(pieces[piece_code])
                piece_image.set("id", f"piece-{piece_code}")
                defs.append(piece_image)

    squares = chess.SquareSet(squares) if squares else chess.SquareSet()
    if squares:
        defs.append(ET.fromstring(XX))

    if check is not None:
        defs.append(ET.fromstring(CHECK_GRADIENT))

    if outer_border:
        outer_border_color, outer_border_opacity = _select_color(colors, "outer border")
        ET.SubElement(svg, "rect", _attrs({
            "x": outer_border / 2,
            "y": outer_border / 2,
            "width": full_size - outer_border,
            "height": full_size - outer_border,
            "fill": "none",
            "stroke": outer_border_color,
            "stroke-width": outer_border,
            "opacity": outer_border_opacity if outer_border_opacity < 1.0 else None,
        }))

    # Render board.
    for square, bb in enumerate(chess.BB_SQUARES):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        x = (file_index if orientation else 7 - file_index) * SQUARE_SIZE + board_offset
        y = (7 - rank_index if orientation else rank_index) * SQUARE_SIZE + board_offset

        cls = ["square", "light" if chess.BB_LIGHT_SQUARES & bb else "dark"]
        if lastmove and square in [lastmove.from_square, lastmove.to_square]:
            cls.append("lastmove")
        square_color, square_opacity = _select_color(colors, " ".join(cls))

        cls.append(chess.SQUARE_NAMES[square])

        ET.SubElement(svg, "rect", _attrs({
            "x": x,
            "y": y,
            "width": SQUARE_SIZE,
            "height": SQUARE_SIZE,
            "class": " ".join(cls),
            "stroke": "none",
            "fill": square_color,
            "opacity": square_opacity if square_opacity < 1.0 else None,
        }))

        try:
            fill_color, fill_opacity = _color(fill[square])
        except KeyError:
            pass
        else:
            ET.SubElement(svg, "rect", _attrs({
                "x": x,
                "y": y,
                "width": SQUARE_SIZE,
                "height": SQUARE_SIZE,
                "stroke": "none",
                "fill": fill_color,
                "opacity": fill_opacity if fill_opacity < 1.0 else None,
            }))

    if coordinates and coordinate_style == "chess.com":
        svg.append(_coordinates(orientation=orientation, style=coordinate_style,
                                colors=colors, offset=board_offset, size=size, full_size=full_size))

    # Render check mark.
    if check is not None:
        file_index = chess.square_file(check)
        rank_index = chess.square_rank(check)

        x = (file_index if orientation else 7 - file_index) * SQUARE_SIZE + board_offset
        y = (7 - rank_index if orientation else rank_index) * SQUARE_SIZE + board_offset

        ET.SubElement(svg, "rect", _attrs({
            "x": x,
            "y": y,
            "width": SQUARE_SIZE,
            "height": SQUARE_SIZE,
            "class": "check",
            "fill": "url(#check_gradient)",
        }))

    # Legal destinations are board-local UI hints. Both source sites render
    # them underneath the piece layer, so capture rings remain visible around
    # the target piece rather than obscuring it.
    for destination_marker in markers:
        cx, cy = _square_center(
            destination_marker.square, orientation=orientation, board_offset=board_offset
        )
        if legal_move_style == "lichess":
            if destination_marker.kind == "dot":
                radius = SQUARE_SIZE * 0.19
                ET.SubElement(svg, "circle", _attrs({
                    "cx": cx,
                    "cy": cy,
                    "r": radius,
                    "fill": "#14551e",
                    "opacity": 0.5,
                    "class": "legal-destination lichess dot",
                }))
                bbox = _box_from_center(cx, cy, radius)
            else:
                x, y = _square_origin(
                    destination_marker.square, orientation=orientation, board_offset=board_offset
                )
                gradient_id = f"legal-capture-{uuid.uuid4().hex}"
                # CSS radial-gradient() defaults to farthest-corner. Match
                # Chessground's 80% stop against that radius, not SVG's 50%
                # object-bounding-box default.
                gradient = ET.SubElement(defs, "radialGradient", _attrs({
                    "id": gradient_id,
                    "gradientUnits": "userSpaceOnUse",
                    "cx": cx,
                    "cy": cy,
                    "r": SQUARE_SIZE / math.sqrt(2),
                }))
                ET.SubElement(gradient, "stop", {
                    "offset": "0%", "stop-color": "#145500", "stop-opacity": "0",
                })
                ET.SubElement(gradient, "stop", {
                    "offset": "80%", "stop-color": "#145500", "stop-opacity": "0",
                })
                ET.SubElement(gradient, "stop", {
                    "offset": "80%", "stop-color": "#145500", "stop-opacity": "0.3",
                })
                ET.SubElement(gradient, "stop", {
                    "offset": "100%", "stop-color": "#145500", "stop-opacity": "0.3",
                })
                ET.SubElement(svg, "rect", _attrs({
                    "x": x,
                    "y": y,
                    "width": SQUARE_SIZE,
                    "height": SQUARE_SIZE,
                    "fill": f"url(#{gradient_id})",
                    "class": "legal-destination lichess capture",
                }))
                bbox = (x, y, x + SQUARE_SIZE, y + SQUARE_SIZE)
        elif destination_marker.kind == "dot":
            # Chess.com's .hint uses 4.2% board-width padding. At eight files
            # this leaves a radius of (1 - 2 * .336) / 2 = .164 squares.
            radius = SQUARE_SIZE * 0.164
            ET.SubElement(svg, "circle", _attrs({
                "cx": cx,
                "cy": cy,
                "r": radius,
                "fill": "#000000",
                "opacity": 0.14,
                "class": "legal-destination chess-com dot",
            }))
            bbox = _box_from_center(cx, cy, radius)
        else:
            # Chess.com starts with a 5px CSS border, then sets borderWidth to
            # clientWidth * .1. clientWidth excludes both borders and rounds to
            # an integer; Chromium then snaps a positive border down to device
            # pixels, with a one-device-pixel minimum.
            css_scale = (css_size if css_size is not None else size or full_size) / full_size
            initial_border = math.floor(5 * device_pixel_ratio) / device_pixel_ratio
            inner_width = max(0, math.floor(SQUARE_SIZE * css_scale - 2 * initial_border + 0.5))
            device_width = max(1, math.floor(inner_width * device_pixel_ratio / 10)) if inner_width else 0
            stroke_width = device_width / device_pixel_ratio / css_scale
            radius = SQUARE_SIZE / 2 - stroke_width / 2
            ET.SubElement(svg, "circle", _attrs({
                "cx": cx,
                "cy": cy,
                "r": radius,
                "fill": "none",
                "stroke": "#000000",
                "stroke-width": stroke_width,
                "opacity": 0.14,
                "class": "legal-destination chess-com capture",
            }))
            bbox = _box_from_center(cx, cy, radius + stroke_width / 2)
        annotations.append(OverlayAnnotation(
            kind="legal_destination_dot" if destination_marker.kind == "dot" else "legal_destination_capture",
            bbox_xyxy=bbox,
        ))

    # Composite each complete ghost piece once, rather than fading its individual SVG paths.
    ghosts = ET.Element("g", {"class": "ghosts", "opacity": "0.3"})
    # Render pieces and selected squares.
    for square, bb in enumerate(chess.BB_SQUARES):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        x = (file_index if orientation else 7 - file_index) * SQUARE_SIZE + board_offset
        y = (7 - rank_index if orientation else rank_index) * SQUARE_SIZE + board_offset

        if board is not None:
            piece = board.piece_at(square)
            if piece:
                if piece_set is None:
                    href = f"#{chess.COLOR_NAMES[piece.color]}-{chess.PIECE_NAMES[piece.piece_type]}"
                else:
                    href = f"#piece-{_piece_code(piece, piece_set=piece_set)}"
                ET.SubElement(ghosts if square in ghost_squares else svg, "use", {
                    "href": href,
                    "xlink:href": href,
                    "transform": f"translate({x:d}, {y:d})",
                })

        # Render selected squares.
        if square in squares:
            ET.SubElement(svg, "use", _attrs({
                "href": "#xx",
                "xlink:href": "#xx",
                "x": x,
                "y": y,
            }))

    lichess_shapes = (
        ET.Element("g", {"class": "shapes lichess", "opacity": "0.6"})
        if any(highlight.palette == "lichess" for highlight in highlights)
        or (arrow_style == "lichess" and bool(arrows))
        else None
    )

    # User highlights are foreground circles, matching Lichess's annotated
    # square geometry. The palette intentionally changes only color, never
    # the class geometry.
    for highlight in highlights:
        cx, cy = _square_center(
            highlight.square, orientation=orientation, board_offset=board_offset
        )
        color_key = f"arrow {highlight.color}"
        palette = CHESS_COM_ARROW_COLORS if highlight.palette == "chess.com" else LICHESS_ARROW_COLORS
        color, opacity = _color(palette[color_key])
        parent = svg if highlight.palette == "chess.com" else lichess_shapes
        assert parent is not None
        # Chessground circleWidth(): a settled circle is 4/64 of a square.
        stroke_width = SQUARE_SIZE * 4 / 64
        radius = SQUARE_SIZE / 2 - stroke_width / 2
        ET.SubElement(parent, "circle", _attrs({
            "cx": cx,
            "cy": cy,
            "r": radius,
            "fill": "none",
            "stroke": color,
            "stroke-width": stroke_width,
            "opacity": opacity if opacity < 1.0 else None,
            "class": f"user-highlight {highlight.palette.replace('.', '-')}",
        }))
        annotations.append(
            OverlayAnnotation(
                kind="user_highlight",
                color=highlight.color,
                bbox_xyxy=_box_from_center(cx, cy, radius + stroke_width / 2),
            )
        )

    if lichess_shapes is not None:
        svg.append(lichess_shapes)

    # Namespace marker IDs so multiple inline boards do not share definitions.
    marker_namespace = uuid.uuid4().hex

    # Render arrows.
    for arrow_index, arrow in enumerate(arrows):
        try:
            tail, head, arrow_color = arrow.tail, arrow.head, arrow.color  # type: ignore
        except AttributeError:
            tail, head = arrow  # type: ignore
            arrow_color = "green"

        color_key = " ".join(["arrow", arrow_color])
        arrow_parent = svg
        if color_key in colors:
            color, opacity = _color(colors[color_key])
        elif arrow_style == "chess.com" and color_key in CHESS_COM_ARROW_COLORS:
            color, opacity = _color(CHESS_COM_ARROW_COLORS[color_key])
        elif arrow_style == "lichess" and color_key in LICHESS_ARROW_COLORS:
            color, opacity = _color(LICHESS_ARROW_COLORS[color_key])
        elif color_key in DEFAULT_COLORS:
            color, opacity = _color(DEFAULT_COLORS[color_key])
        else:
            color, opacity = _color(arrow_color)
        if arrow_style == "lichess":
            if not _is_valid_color(color, opacity, require_opaque=True):
                raise ValueError("lichess arrow colors must be opaque hex or named colors")
            assert lichess_shapes is not None
            arrow_parent = lichess_shapes
        elif not _is_valid_color(color, opacity, require_opaque=False):
            raise ValueError("chess.com arrow colors must be visible hex or named colors")

        tail_file = chess.square_file(tail)
        tail_rank = chess.square_rank(tail)
        head_file = chess.square_file(head)
        head_rank = chess.square_rank(head)

        xtail = board_offset + (tail_file + 0.5 if orientation else 7.5 - tail_file) * SQUARE_SIZE
        ytail = board_offset + (7.5 - tail_rank if orientation else tail_rank + 0.5) * SQUARE_SIZE
        xhead = board_offset + (head_file + 0.5 if orientation else 7.5 - head_file) * SQUARE_SIZE
        yhead = board_offset + (7.5 - head_rank if orientation else head_rank + 0.5) * SQUARE_SIZE

        if (head_file, head_rank) == (tail_file, tail_rank):
            arrow_direction = (1.0, 0.0)
            radius = SQUARE_SIZE * 0.5
            stroke_width = SQUARE_SIZE * (4 / 64 if arrow_style == "lichess" else 0.1)
            ET.SubElement(arrow_parent, "circle", _attrs({
                "cx": xhead,
                "cy": yhead,
                "r": radius - stroke_width / 2,
                "stroke-width": stroke_width,
                "stroke": color,
                "opacity": opacity if opacity < 1.0 else None,
                "fill": "none",
                "class": "circle",
            }))
            primitive_points = (
                (xhead - radius, yhead - radius),
                (xhead + radius, yhead - radius),
                (xhead + radius, yhead + radius),
                (xhead - radius, yhead + radius),
            )
            primitive_bbox = _points_bbox(primitive_points)
            arrowhead_bbox = None
            painted_tail = (xtail, ytail)
            painted_head = (xhead, yhead)
        elif arrow_style == "lichess":
            marker_id = f"arrowhead-{marker_namespace}-{arrow_index}"
            marker = ET.SubElement(defs, "marker", {
                "id": marker_id,
                "orient": "auto",
                "overflow": "visible",
                "markerWidth": "4",
                "markerHeight": "4",
                "refX": "2.05",
                "refY": "2",
            })
            ET.SubElement(marker, "path", {
                "d": "M0,0 V4 L3,2 Z",
                "fill": color,
                "class": "arrow lichess",
            })

            dx, dy = xhead - xtail, yhead - ytail
            arrow_direction = (dx, dy)
            hypot = math.hypot(dx, dy)
            margin = SQUARE_SIZE * 10 / 64
            ux, uy = dx / hypot, dy / hypot
            px, py = -uy, ux
            line_end_x = xhead - ux * margin
            line_end_y = yhead - uy * margin

            ET.SubElement(arrow_parent, "line", _attrs({
                "x1": xtail,
                "y1": ytail,
                "x2": line_end_x,
                "y2": line_end_y,
                "stroke": color,
                "opacity": opacity if opacity < 1.0 else None,
                "stroke-width": SQUARE_SIZE * 10 / 64,
                "stroke-linecap": "round",
                "marker-end": f"url(#{marker_id})",
                "class": "arrow lichess",
            }))
            half_stroke = margin / 2
            marker_points = (
                (line_end_x - ux * 2.05 * margin - px * 2 * margin,
                 line_end_y - uy * 2.05 * margin - py * 2 * margin),
                (line_end_x - ux * 2.05 * margin + px * 2 * margin,
                 line_end_y - uy * 2.05 * margin + py * 2 * margin),
                (line_end_x + ux * 0.95 * margin,
                 line_end_y + uy * 0.95 * margin),
            )
            painted_tail = (xtail - ux * half_stroke, ytail - uy * half_stroke)
            painted_head = marker_points[2]
            primitive_points = (
                (xtail - ux * half_stroke - px * half_stroke,
                 ytail - uy * half_stroke - py * half_stroke),
                (xtail - ux * half_stroke + px * half_stroke,
                 ytail - uy * half_stroke + py * half_stroke),
                (line_end_x + ux * half_stroke - px * half_stroke,
                 line_end_y + uy * half_stroke - py * half_stroke),
                (line_end_x + ux * half_stroke + px * half_stroke,
                 line_end_y + uy * half_stroke + py * half_stroke),
            ) + marker_points
            marker_bbox = _points_bbox(marker_points)
            arrowhead_bbox = marker_bbox
            primitive_bbox = (
                min(min(xtail, line_end_x) - half_stroke, marker_bbox[0]),
                min(min(ytail, line_end_y) - half_stroke, marker_bbox[1]),
                max(max(xtail, line_end_x) + half_stroke, marker_bbox[2]),
                max(max(ytail, line_end_y) + half_stroke, marker_bbox[3]),
            )
        else:
            tail_gap = SQUARE_SIZE * 0.36
            shaft_half_width = SQUARE_SIZE * 0.11
            head_length = SQUARE_SIZE * 0.36
            head_half_width = SQUARE_SIZE * 0.26

            dx, dy = xhead - xtail, yhead - ytail
            is_knight_move = (abs(head_file - tail_file), abs(head_rank - tail_rank)) in [(1, 2), (2, 1)]
            arrow_direction = (1.0, 0.0) if is_knight_move else (dx, dy)
            if is_knight_move:
                if abs(dx) > abs(dy):
                    ux, uy = math.copysign(1.0, dx), 0.0
                    vx, vy = 0.0, math.copysign(1.0, dy)
                else:
                    ux, uy = 0.0, math.copysign(1.0, dy)
                    vx, vy = math.copysign(1.0, dx), 0.0

                major_length = 2 * SQUARE_SIZE
                points = [
                    (xtail + ux * tail_gap - vx * shaft_half_width,
                     ytail + uy * tail_gap - vy * shaft_half_width),
                    (xtail + ux * (major_length + shaft_half_width) - vx * shaft_half_width,
                     ytail + uy * (major_length + shaft_half_width) - vy * shaft_half_width),
                    (xhead - vx * head_length + ux * shaft_half_width,
                     yhead - vy * head_length + uy * shaft_half_width),
                    (xhead - vx * head_length + ux * head_half_width,
                     yhead - vy * head_length + uy * head_half_width),
                    (xhead, yhead),
                    (xhead - vx * head_length - ux * head_half_width,
                     yhead - vy * head_length - uy * head_half_width),
                    (xhead - vx * head_length - ux * shaft_half_width,
                     yhead - vy * head_length - uy * shaft_half_width),
                    (xtail + ux * (major_length - shaft_half_width) + vx * shaft_half_width,
                     ytail + uy * (major_length - shaft_half_width) + vy * shaft_half_width),
                    (xtail + ux * tail_gap + vx * shaft_half_width,
                     ytail + uy * tail_gap + vy * shaft_half_width),
                ]
            else:
                hypot = math.hypot(dx, dy)
                ux, uy = dx / hypot, dy / hypot
                px, py = -uy, ux
                shaft_start_x = xtail + ux * tail_gap
                shaft_start_y = ytail + uy * tail_gap
                arrowhead_x = xhead - ux * head_length
                arrowhead_y = yhead - uy * head_length

                points = [
                    (shaft_start_x + px * shaft_half_width, shaft_start_y + py * shaft_half_width),
                    (arrowhead_x + px * shaft_half_width, arrowhead_y + py * shaft_half_width),
                    (arrowhead_x + px * head_half_width, arrowhead_y + py * head_half_width),
                    (xhead, yhead),
                    (arrowhead_x - px * head_half_width, arrowhead_y - py * head_half_width),
                    (arrowhead_x - px * shaft_half_width, arrowhead_y - py * shaft_half_width),
                    (shaft_start_x - px * shaft_half_width, shaft_start_y - py * shaft_half_width),
                ]

            ET.SubElement(arrow_parent, "polygon", _attrs({
                "points": " ".join(f"{x:g},{y:g}" for x, y in points),
                "fill": color,
                "opacity": opacity if opacity < 1.0 else None,
                "class": "arrow chess-com",
            }))
            primitive_points = tuple(points)
            primitive_bbox = _points_bbox(primitive_points)
            arrowhead_bbox = _points_bbox(points[3:6] if is_knight_move else points[2:5])
            painted_tail = (
                (points[0][0] + points[-1][0]) / 2,
                (points[0][1] + points[-1][1]) / 2,
            )
            painted_head = points[4] if is_knight_move else points[3]

        annotation_color: Optional[CanonicalOverlayColor] = (
            arrow_color if arrow_color in {"green", "red", "yellow", "blue"} else None
        )
        arrow_obb = _oriented_box(primitive_points, arrow_direction)
        annotations.append(
            OverlayAnnotation(
                kind="arrow",
                color=annotation_color,
                bbox_xyxy=primitive_bbox,
                arrowhead_bbox_xyxy=arrowhead_bbox,
                tail_xy=painted_tail,
                head_xy=painted_head,
                obb_xyxyxyxy=arrow_obb,
            )
        )

    if len(ghosts):
        svg.append(ghosts)

    if coordinates and coordinate_style == "lichess":
        svg.append(_coordinates(orientation=orientation, style=coordinate_style,
                                colors=colors, offset=board_offset, size=size, full_size=full_size))

    return BoardRenderResult(
        svg=SvgWrapper(ET.tostring(svg).decode("utf-8")),
        viewbox_size=full_size,
        annotations=tuple(annotations),
    )
