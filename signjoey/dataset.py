# coding: utf-8
"""
Data module
"""
import numpy as np
from collections import OrderedDict
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch


def load_dataset_file(filename):
    # with gzip.open(, "rb") as f:
    with open(filename, "rb") as f:
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
                ("gls", fields[2]),
                ("txt", fields[3]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
<<<<<<< HEAD
                seq_label = s["label"]

                #ordered_signs = OrderedDict(sorted(s["signs"].items(), key=lambda x: int(x[0])))
                #res = np.array(list(ordered_signs.values()))
                s["signs"] = torch.Tensor(s["signs"].squeeze())
=======
                if not s.get("labels") and not s.get("label"):
                    raise KeyError("Key 'labels' or 'label' not found.")

                seq_label = s.get("labels") or s.get("label")

                if isinstance(s["signs"], dict):
                    ordered_signs = OrderedDict(sorted(s["signs"].items(), key=lambda x: int(x[0])))
                    res = np.array(list(ordered_signs.values()))
                else:
                    res = s["signs"]
                s["signs"] = torch.Tensor(res.squeeze())
>>>>>>> f5696d1bb95885f180c54966c9a4b20fdcff835b
                if seq_label in samples:
                    raise Exception(f"Label {seq_label} already exists.")
                else:
                    if s["signs"].shape[0]:
<<<<<<< HEAD
                       samples[seq_label] = {
                           "label": s["label"],
                           "gloss": s["verse"],
                           "text": s["verse"],
                           "sign": s["signs"],
                       }
=======
                        samples[seq_label] = {
                            "label": s.get("labels") or s.get("label"),
                            "gloss": s["verse"],
                            "text": s["verse"],
                            "sign": s["signs"],
                        }
>>>>>>> f5696d1bb95885f180c54966c9a4b20fdcff835b

        examples = []
        for s in samples:
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["label"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
