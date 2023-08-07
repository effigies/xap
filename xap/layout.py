import os
import re
import typing as ty
from functools import cached_property
from pathlib import Path

from bids2table import bids2table

import bidsschematools as bst  # type: ignore[import]
import bidsschematools.schema  # type: ignore[import]
import bidsschematools.types  # type: ignore[import]

from . import types as bt


class BIDSValidationError(ValueError):
    """Error arising from invalid files or values in a BIDS dataset"""


class Schema:
    schema: bst.types.Namespace

    def __init__(
        self,
        schema: ty.Union[bst.types.Namespace, None] = None,
    ):
        if schema is None:
            # Bundled
            schema = bst.schema.load_schema()
        self.schema = schema

    @classmethod
    def from_spec(cls, schema_spec: str) -> "Schema":
        return cls(bst.schema.load_schema(schema_spec))

    # Conveniences to avoid `schema.schema` pattern
    @property
    def objects(self) -> bst.types.Namespace:
        return self.schema.objects

    @property
    def rules(self) -> bst.types.Namespace:
        return self.schema.rules

    @property
    def meta(self) -> bst.types.Namespace:
        return self.schema.meta


default_schema = Schema()


class File(bt.File[Schema]):
    """Generic file holder

    This serves as a base class for :class:`BIDSFile` and can represent
    non-BIDS files.
    """

    def __init__(
        self,
        path: ty.Union[os.PathLike, str],
        dataset: ty.Optional["BIDSDataset"] = None,
    ):
        self.path = Path(path)
        self.dataset = dataset


class BIDSFile(File, bt.BIDSFile[Schema]):
    """BIDS file"""

    def __init__(
        self,
        path: ty.Union[os.PathLike, str],
        dataset: ty.Optional["BIDSDataset"] = None,
        *,
        entities: Optional[Dict[str, Union[Label, Index]]] = None,
        datatype: Optional[str] = None,
        suffix: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        super().__init__(path, dataset)
        self.entities = entities or {}
        self.datatype = datatype
        self.suffix = suffix
        self.extension = extension

        schema = default_schema if dataset is None else dataset.schema

    @classmethod
    def from_row(cls, row: pd.Series, dataset: "BIDSDataset") -> "BIDSFile":
        entities = {
            col[5:]: row[col]
            for col in row.dropna().index
            if col.startswith('ent__') and col[5:] not in ('datatype', 'suffix', 'ext', 'extra_entities')
        }
        return cls(
            row['file__file_path'],
            self,
            entities=entities,
            datatype=row['ent_datatype'],
            suffix=row['suffix'],
            extension=row['ent_ext'],
        )

    @cached_property
    def metadata(self) -> dict[str, ty.Any]:
        """Sidecar metadata aggregated according to inheritance principle"""
        if not self.dataset:
            raise ValueError
        return dataset.table['meta__json'][dataset.table['file__file_path'] == self.path].item()


class BIDSDataset(bt.BIDSDataset[Schema]):
    def __init__(self, root: ty.Union[os.PathLike, str], **kwargs):
        self.schema = kwargs.pop('schema', default_schema)
        self.table = bids2table(root, **kwargs)

        self.dataset_description = self.table['ds__dataset_description'][0]
        self.files = [BIDSFile.from_row(row, self) for index, row in self.table.iterrows()]
        self.datatypes = self.table.ent__datatype.unique().tolist()
        self.subjects = self.table.ent__sub.unique().tolist()

        self.ignored = []  # TODO
        self.modalities = []  # TODO
        self.entities = []  # TODO
