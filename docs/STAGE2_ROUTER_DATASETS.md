# Stage 2 Router Datasets

`prepare_data.py` downloads datasets through Hugging Face, verifies each schema and dataset-card license, caps each class at 2,500, removes within-class duplicates, adds neutral task queries, and writes deterministic 70/15/15 splits under ignored `.tools/stage2/data/router/`. Seed `20260905` produced 17,641 examples: 12,274 train, 2,662 validation, and 2,705 test. Upstream official splits were not combined because capped class filtering is required across sources; content-hash splitting prevents the same normalized sample crossing splits.

| Route | Public source | Revision | License | Total |
|---|---|---:|---|---:|
| text/biomedical | qiaojin/PubMedQA, pqa_labeled | `9001f28` | MIT | 1,000 |
| text/legal | coastalcph/lex_glue, ledgar | `c23fdff` | CC BY 4.0 | 2,500 |
| text/finance | zeroshot/twitter-financial-news-sentiment | `ccbe24d` | MIT | 2,500 |
| code/python | CodeSearchNet Python | `bd0cf26` | dataset card: other | 2,500 |
| code/java | CodeSearchNet Java | `bd0cf26` | dataset card: other | 2,500 |
| code/sql | b-mc2/sql-create-context | `9d80a6a` | CC BY 4.0 | 2,500 |
| math/algebra | MATH-lighteval | `0530c78` | MIT | 2,500 |
| math/geometry | MATH-lighteval | `0530c78` | MIT | 870 |
| math/probability | MATH-lighteval | `0530c78` | MIT | 771 |

PubMedQA and the two smaller MATH categories are below the 2,000 target because the selected public configurations contain fewer qualifying unique examples. FinancialPhraseBank and WikiSQL were inspected first, but their legacy dataset scripts are unsupported by the installed `datasets` runtime. The pipeline uses parquet-backed substitutes with declared licenses instead.

Natural questions are retained where present. Other samples use a fixed neutral template set such as “Explain this,” “Why is this failing,” and “Solve this.” Templates never name the target domain. `source` is retained only in ignored JSONL for audit and is excluded from training features.
