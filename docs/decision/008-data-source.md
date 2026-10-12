# 008: Real Training Data Sources (Phase 7a)

Status: **Proposed**. Nothing downloaded yet. Checked 2026-10-11 against
the Common Pile paper (arXiv 2506.05209), the Comma dataset card on
Hugging Face and the EleutherAI announcement. Exact subset names, sizes
and per-card license text are re-checked when the manifest is built
(Phase 7b).

## Decisions
1. Source: Common Pile v0.1 (filtered per-source datasets) only.
2. Excluded: FineWeb and FineWeb-Edu (the ODC-By collection license does
   not license individual documents), LLM-generated text, and any
   NC/ND or unclear-license source.
3. Excluded from base pretraining: Data Provenance Initiative (DPI)
   task data. It contains Winogrande, a planned benchmark. It may be
   used for instruction tuning after decontamination.
4. Mix v1 below. Provisional: tested with small proxies before the run.

## Why the Common Pile
- About 8 TB from 30 sources, all public domain or licensed under terms
  meeting the Open Definition 2.1. NC and ND licenses excluded.
- Sources with unreliable license metadata were excluded by its authors
  (OpenAlex, YouTube Commons, a Kaggle Hacker News dump), and LLM-generated
  text was avoided.
- Filtered and deduplicated in the Comma dataset: language ID, web quality,
  OCR likelihood, toxicity, regex PII removal (emails, phones, IPs), fuzzy
  dedup (over 90% shared 20-grams).
- Validated by 7B models (Comma v0.1) competitive with budget-matched
  models trained on unlicensed data.

## Mix v1 (5.0B tokens, no source repeated)
| Group | Common Pile sources | Share | Tokens |
|---|---|---|---|
| General web, spoken, educational | CC web pages, open textbooks, CC-BY YouTube transcripts, CC news | 30% | 1.50B |
| Q&A and discussion | StackExchange, some GitHub threads, Ubuntu IRC | 16% | 0.80B |
| Reference | Wikimedia wikis | 14% | 0.70B |
| Science and scholarly | peS2o, arXiv, PubMed | 11% | 0.55B |
| Code | Stack v2 educational-quality subset | 15% | 0.75B |
| Books | Pre-1929 books, Gutenberg, Library of Congress | 9% | 0.45B |
| Reserve | filled after first evaluation | 5% | 0.25B |

Availability (Comma tokenizer, billions): web 15.2, peS2o 43.3,
StackExchange 23.9, Stack v2 edu 67.8, Wikimedia 15.8, pre-1929 books
about 11.6. Sizes of textbooks, YouTube, news, IRC, PubMed, arXiv: to be
measured in 7b. Our 16k tokenizer yields more tokens per byte, so real
availability is higher.

## Our processing on top of the Common Pile filtering
- Decontamination against every eval set (13-gram overlap), including
  our custom math and coding tests.
- Secret/API-key scan for code and a second PII pass.
- Per-source provenance and license record in data/manifest.yaml.
- Interleave and shuffle into 100M-token shards so that any prefix
  (for example the first 1B tokens) approximates the full mix.

## Verified dataset names (Hugging Face, org `common-pile`)
- Raw: 30 datasets, e.g. cccc, stackexchange, wikimedia, peS2o, stackv2,
  pre_1929_books, project_gutenberg, library_of_congress, libretexts,
  oercommons, pressbooks, doab, news, youtube, pubmed, arxiv_papers,
  arxiv_abstracts, ubuntu_irc, github_archive.
- Filtered: same names with the suffix `_filtered` (31 items; the Stack v2
  split is resolved at run time). The sampler prefers `_filtered` and
  reports when it falls back to a raw dataset.
- Files are gzipped JSON shards (about 800 MB each) with fields id, text,
  source, added, metadata. Some documents are 16M characters long.
- Card caveats: some cards list effective (post-repeat) token counts, so we
  count our own. The public filtered cccc may differ from the exact version
  Comma used. Every card warns about license laundering.
- Revisions are pinned in data/manifest.yaml (section common_pile).


## Known weaknesses
- Weaker on everyday commonsense and casual chat than web-trained models
  (published HellaSwag/PIQA/CommonsenseQA results); stronger on science.
- Old-book content skews archaic and carries period biases.
- English only.
- License metadata dates from late 2024 and can be wrong; CC BY-SA and
  attribution terms apply; the legal status of weights trained on
  share-alike data is unsettled. Not legal advice.

## Open items (Phase 7b)
- Exact Hugging Face subset names and license text per card.
- Sizes of small sources; adjust shares if one is too small.
- Whether a 5% reserve or a stronger science/math share is better.

## Sources
- https://arxiv.org/abs/2506.05209
- https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset
- https://blog.eleuther.ai/common-pile/
- https://huggingface.co/common-pile


## Findings from the first sampling run (7b.1, 2026-10-11)
- All 21 sources resolved to `_filtered` datasets and streamed correctly.
- 20 of 21 sources carry a per-document license field: CC BY, CC BY-SA,
  CC0, public domain, MIT, Apache-2.0, BSD-2/3-Clause, Unlicense. No NC or
  ND values in the 300-document samples.
- cccc has no per-document license; its provenance is the domain-level
  verification on its dataset card (537 domains). Kept, caveat recorded.
- arxiv_abstracts: the `license` field (CC0) is the abstract's license.
  `full_text_license` describes the full paper and is ignored.
- peS2o stores its license in `oa_license`. stackv2_edu: use `license`
  (`gha_license_id` is often missing). github_archive license strings
  differ in case and are normalized.
- wikimedia: the sample was mostly Wiktionary-style stubs with broken
  template text (median 173 characters). Needs a project filter and a
  minimum length. Pending the 7b.2 audit.
- Each sample came from a single shard, so sizes are not representative.