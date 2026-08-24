# OGDL Attribution and Redistribution Audit

## Conclusion

**GO for repository/source-archive redistribution of the two files under `data/sample/`, provided this attribution is retained.** The relevant government dataset is expressly published under Taiwan's Government Open Data License, version 1.0 (OGDL 1.0). OGDL grants reproduction, distribution, public transmission, editing/adaptation, and sublicensing without a separate written permission, while making attribution a condition of the license.

This is a repository licensing audit, not legal advice. The conclusion is limited to the identified Ministry of Justice dataset and these two normalized snapshots.

## Official source and license

- Data providing organization: **法務部資訊處** (Ministry of Justice, Department of Information Management)
- Dataset: **[中文法規_命令資料檔下載](https://data.gov.tw/dataset/18290)** (dataset ID 18290)
- Dataset page authorization: **政府資料開放授權條款－第1版**
- License text: **[政府資料開放授權條款－第1版 / Open Government Data License 1.0](https://data.gov.tw/license)**
- Dataset resource type: original monthly file data, free of charge

The dataset page identifies the same fields contained in the samples, including legal nature, name, URL, category, latest amendment, effective date, history, article number, and article content.

## Retained attribution statement

> 法務部資訊處（2026 repository snapshot），「中文法規_命令資料檔下載」。此開放資料依政府資料開放授權條款 (Open Government Data License) 第 1 版進行公眾釋出，使用者於遵守該條款各項規定之前提下，得利用之。授權條款：https://data.gov.tw/license

“2026 repository snapshot” distinguishes when these normalized copies entered this project; it does not assert a separate Ministry of Justice dataset version number. Each file's own `last_amended` field remains the authoritative snapshot discriminator below.

## Distributed snapshots

Hashes use the repository's canonical UTF-8/LF text representation, so checkout line-ending settings do not change identity.

| File | Official law page recorded in file | `last_amended` | Canonical SHA-256 |
|---|---|---:|---|
| `data/sample/勞工請假規則.json` | [勞工請假規則](https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030006) | 2025-12-09 | `f321a5beaf23521bb572a33bf5d87d3ec4241554cd75880c9fe4bd2d7d9b3abb` |
| `data/sample/勞動基準法施行細則.json` | [勞動基準法施行細則](https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0030002) | 2024-03-27 | `99b5e4756f5f8ed53286d620ebadaefa6dc8f37e00cf62014fa8d05ff29a5e78` |

`scripts/download_corpus.py` shows the transformation provenance: it downloads the official command XML dump, selects the two target regulations, normalizes them into JSON, and copies them into `data/sample/`.

## Rights and obligations applied here

OGDL 1.0 section 2 permits use for any purpose and expressly includes reproduction, distribution, public transmission, compilation, editing, and adaptation. It also permits sublicensing and does not require separate written permission. This covers storing the normalized JSON samples in a Git repository and including them in a source archive.

OGDL 1.0 section 3 requires a clear attribution to the original data providing organization and release. Failure to satisfy that attribution condition means the license grant does not apply. The attribution in this file and the links in both READMEs therefore form part of the public release boundary.

OGDL does not grant patent or trademark rights and includes source-data warranty/liability disclaimers. Redistribution does not imply Ministry of Justice endorsement.

## Relationship to MIT

The repository's original software is licensed under [MIT](../../LICENSE). The two sample JSON files remain under OGDL 1.0 and are not relicensed under MIT. The full downloaded dump and the other 13 normalized target files are excluded from the repository/publication allowlist; their absence is a project distribution choice, not an OGDL prohibition.

This resolves the former README ambiguity: the **full corpus is not distributed**, while **two attributed OGDL samples are distributed** for offline smoke tests.
