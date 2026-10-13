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

## Audit findings (7b.2, 2026-10-12)
Sampled about 3,990 documents across 21 sources and many shards each.
- Licenses: 6 documents (0.15%) fail the allow-list (1 WTFPL, 5 Stack v2
  files with empty, BSD-4-Clause, BSD-2-Clause-Views or UPL-1.0+ values).
  They are dropped. No NC or ND values anywhere. arxiv_abstracts: the
  `license` field (CC0) is used, not `full_text_license`.
- wikimedia is a mixture: about 10% Wiktionary, only 62% namespace 0,
  31% under 500 characters. Keep wikipedia.com + namespace 0 only.
- Documents are ordered inside shards (one book, one Stack Exchange
  site, one programming language per shard). The builder samples
  uniformly across shards and enforces per-stratum limits.
- Short or noisy: github_archive (32% under 500 characters), ubuntu_irc
  (14%), OER sources (10-13%). A 500-character minimum applies.
- No replacement-character corruption in any source.

## Filter rules v1 (implemented in src/vibe/filters.py)
License allow-list (cccc verified by domain); minimum 500 characters (300
for abstracts and code); no replacement characters; letters at least 50%
for prose sources; English Wikipedia articles only; drop generated and
vendored code; Stack Overflow at most 35% of the Stack Exchange budget;
code language targets Python 28, JavaScript 14, TypeScript 8, Java 8,
C 7, C++ 7, Go 5, Rust 5, SQL 4, PHP 4, C# 3, Ruby 2, Shell 2,
Markdown 2, Swift 1 (percent; names verified in the next scan).

## Revisions seen in the audit (first 10 characters of the commit)
cccc 03a3de5713, youtube dff8c8a54e, libretexts 70388bca52, oercommons
506b6159da, pressbooks 1a1d3b50d7, doab defb24ca72, news 59aaa8f104,
foodista bf2c7aaa2d, public_domain_review efc7f21259, stackexchange
c0ac737383, github_archive 52282fe966, ubuntu_irc 84f88c9865, wikimedia
0641bb84bd, pes2o 297747513b, arxiv_papers 033cf7f53f, arxiv_abstracts
dc1ceab475, pubmed c156f0569a, stackv2_edu c354dbe884, pre_1929_books
23f9d96dbb, project_gutenberg 3cdf6879c8, library_of_congress 56725c7aa1.
The builder records the full hashes in the corpus metadata.