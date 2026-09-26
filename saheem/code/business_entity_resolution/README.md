# Business entity resolution pipeline

The package under `src/entity_resolution` is the self-contained implementation used
to reproduce candidate generation, model training, and predictions. At the current
checkpoint it provides the data contracts, normalizers, split logic, and evaluator.

From the repository root, run commands with:

```bash
PYTHONPATH=code/business_entity_resolution/src python -m entity_resolution --help
```

Only the supplied competition files may be used for entity resolution. External
business lookup, geocoding, external databases, and internet data augmentation are
not part of this pipeline.
