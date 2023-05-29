# coding: utf-8
"""
Data module
"""
import numpy as np
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch


def load_dataset_file(filename):
    import os
    print(os.getcwd())
    print(filename)
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("sgn", fields[1]),
                ("vrs_enc", fields[2]),
                ("vrs_dec", fields[3]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_label = s["label"]

                res = np.array(list(s["signs"].values()))
                s["signs"] = torch.Tensor(res.squeeze())
                if seq_label in samples:
                    raise Exception(f"Label {seq_label} already exists.")
                else:
                    samples[seq_label] = {
                        "label": s["label"],
                        "verse_enc": s["verse"],
                        "verse_dec": s["verse"],
                        "sign": s["signs"],
                    }

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["label"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["verse_enc"].strip(),
                        sample["verse_dec"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
