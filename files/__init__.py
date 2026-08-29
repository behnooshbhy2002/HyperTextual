"""
hyper_text
==========

Extension package that adds textual/semantic information about entities and
relations to the original HYPER model, WITHOUT modifying any file under
`hyper/` or `script/run.py`.

Everything here either:
  1) imports and reuses unmodified classes/functions from `hyper.*`, or
  2) subclasses them and overrides only the methods that need the new
     "text injection" logic (the overridden methods are literal copies of
     the parent method with the new lines clearly marked with
     `# >>> TEXT ... # <<< TEXT` so it's easy to diff against upstream
     if `hyper/models.py` changes in the future).

Modules
-------
text_encoder.py   -> builds/caches semantic embeddings for entities/relations
text_datasets.py  -> loads a HYPER dataset AND attaches text embeddings to it
text_models.py    -> TextRelHCNet, TextEntityHCNet, TextHYPER, TextFusion
"""
