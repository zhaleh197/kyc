"""Document profiles: everything country- and layout-specific lives here.

The pipeline itself knows nothing about Iran. It asks a profile which fields
exist, which detector class maps to which field, how to normalise a value and
how to validate the set. Adding the Iraqi national card later means adding one
profile module plus a trained detector - no pipeline change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from kyc.core.schemas import Reason


class FieldKind(str, Enum):
    TEXT = "text"          # Persian/Arabic free text -> normalize_name
    DIGITS = "digits"      # numeric only -> keep_digits
    DATE = "date"          # Jalali/Gregorian date -> normalize_date
    CODE = "code"          # latin letters + digits (card number, blood group) -> normalize_code
    IMAGE = "image"        # portrait or signature crop, not OCR'd
    CONTAINER = "container"  # the card itself; used for rectification, not a value


@dataclass(frozen=True)
class FieldSpec:
    key: str                 # canonical, language-neutral key used in the API
    detector_class: str      # class name emitted by the trained field detector
    kind: FieldKind
    label_fa: str = ""
    label_en: str = ""
    required: bool = True
    exact_len: int | None = None   # e.g. 10 for the national id
    charset_hint: str | None = None  # passed to recognisers that support an allowlist
    lexicon: str | None = None  # closed vocabulary in ocr/lexicon/<name>.txt
    # When the crop matches nothing on the list, try each of its text lines:
    # for a short list no neighbouring row can be on (sex), never for names -
    # the row above a name is often another valid name.
    lexicon_per_line: bool = False


@dataclass(frozen=True)
class DocumentProfile:
    id: str
    country: str             # ISO 3166-1 alpha-2
    name_en: str
    name_fa: str
    detector_model: str      # filename inside models/
    fields: tuple[FieldSpec, ...]
    # Class order the detector was TRAINED with. The ONNX output gives class
    # indices, so this list is the index->name mapping and must be kept in sync
    # with the training config - not derived from `fields`, whose order is only
    # a presentation concern.
    class_names: tuple[str, ...] = ()
    # Physical aspect ratio of the document, used to rectify the card to a
    # canonical size before field detection. ID-1 (credit-card) is 85.6x53.98mm.
    # Spatial rules that hold on an upright card, (detector class, relation,
    # detector class) with relation "left_of" or "above". Used to find the
    # card's orientation when it is photographed sideways or upside down.
    layout: tuple[tuple[str, str, str], ...] = ()
    aspect_ratio: float = 85.6 / 53.98
    # Recogniser backends for this document; None = the configured default
    # (settings.ocr.text_backend / digit_backend). digit_backend also reads
    # dates and latin codes.
    text_backend: str | None = None
    digit_backend: str | None = None
    # Flatten the card before detection; None = settings.ocr.rectify. Must
    # match how the profile's detector was trained (whole photos or flat cards).
    rectify: bool | None = None
    # Arabic letters kept in names instead of folded onto Persian ones (ة -> ه
    # and so on suits Iranian cards; Arabic and Kurdish names need them).
    keep_letters: str = ""
    # Cut a label's colon off the right edge of text crops (see
    # pipeline.trim_label_colon); for cards that print "label: value".
    trim_label_colon: bool = False
    # Cut the edges of the rows above and below off text crops (see
    # pipeline.drop_neighbour_rows); for cards whose rows sit close.
    drop_neighbour_rows: bool = False
    # Detector classes of the value rows, top to bottom, for a card that
    # prints them one under the other at an even pitch: a box is put on the
    # row it sits on whatever its label (see rows.place_rows), and an empty
    # row is read where it would be.
    row_order: tuple[str, ...] = ()
    # Read each text crop a second time, lightly blurred, and keep the surer
    # reading: a sharp security print behind the letters (fine diagonal
    # lines) can make the recogniser drop or change a letter.
    reread_blurred: bool = False
    rectified_width: int = 1024
    validator: Callable[[dict], list[Reason]] | None = field(default=None, compare=False)
    # Optional post-processing that adds derived values (gregorian dates,
    # computed age, checksum verdict). Kept on the profile so the pipeline
    # never has to import a country module.
    enricher: Callable[[dict], dict] | None = field(default=None, compare=False)

    @property
    def rectified_size(self) -> tuple[int, int]:
        return self.rectified_width, int(round(self.rectified_width / self.aspect_ratio))

    @property
    def container_class(self) -> str | None:
        for spec in self.fields:
            if spec.kind is FieldKind.CONTAINER:
                return spec.detector_class
        return None

    @property
    def value_fields(self) -> tuple[FieldSpec, ...]:
        """Fields that carry a readable value (everything but image/container)."""
        return tuple(f for f in self.fields if f.kind not in (FieldKind.IMAGE, FieldKind.CONTAINER))

    def by_detector_class(self, name: str) -> FieldSpec | None:
        for spec in self.fields:
            if spec.detector_class == name:
                return spec
        return None

    def by_key(self, key: str) -> FieldSpec | None:
        for spec in self.fields:
            if spec.key == key:
                return spec
        return None

    def class_name(self, index: int) -> str | None:
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return None

    def validate(self, values: dict) -> list[Reason]:
        return self.validator(values) if self.validator else []

    def enrich(self, values: dict) -> dict:
        return self.enricher(values) if self.enricher else {}
