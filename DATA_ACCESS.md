# Data Access

This repository is clean-room and public-safe by default. Its examples, tests,
and continuous-integration jobs use deterministic synthetic reviews only.

## Optional real-data benchmark

The private benchmark was developed against the Amazon Fine Food Reviews
dataset described by Stanford SNAP:

- Kaggle listing: <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews>
- Stanford SNAP source page: <https://snap.stanford.edu/data/web-FineFoods.html>

The Kaggle listing currently labels the dataset **CC0: Public Domain**. The
Stanford SNAP source page provides provenance, schema, a download, and a
research citation, but does not state a separate dataset license.

Because the records contain third-party review text and linkable user and
product fields, this repository does not represent that every underlying right
has been cleared for redistribution. CC0 applies only to rights the affirmer is
authorized to waive and does not clear third-party copyright, privacy,
publicity, or trademark rights.

If you choose to conduct a separate private benchmark:

1. Review the source pages, current license metadata, and applicable terms.
2. Obtain the data directly from a source you determine is authorized for your
   intended use.
3. Store it outside the repository.
4. Build or use a private adapter outside this public repository; the current
   CLI intentionally supports synthetic data only.
5. Do not commit raw rows, review excerpts, identifiers, fitted artifacts, or
   generated caches.

The project does not download the source dataset automatically, and real data
is never required for tests or CI. Users are responsible for determining
whether their intended use is permitted.

This document is informational and is not legal advice.
