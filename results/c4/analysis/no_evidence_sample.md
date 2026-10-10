# Layer 1b -- Dual-Rater Blind Human Audit of the `no_evidence` Heuristic Label (50 Spans)

Fifty snippets sampled with `seed = 42` that the phrase list called 'no evidence exists'.
Each snippet was independently audited by Rater 1 (E. M. S. N. Rathnasiri, IT22047724; 47/50 = 94.0%)
and Rater 2 (blind second team auditor; 46/50 = 92.0%), yielding 98.0% inter-annotator agreement
and **Cohen's kappa = 0.8466** (`results/c4/analysis/evidence_audit.json`).

| # | response_id | split | task | label type | annotator's Original: text | rater 1 | rater 2 |
|---|---|---|---|---|---|---|---|
| 1 | 58 | test | summarization | Evident Baseless Info | This generated details is not mentioned in source content. | yes | yes |
| 2 | 404 | calib | summarization | Evident Baseless Info | No mentioned of allowing doctors to practice medicine exams. However, source content mentioned that the traiditional way to become the doctor is to pass national exams. | yes | yes |
| 3 | 596 | calib | summarization | Evident Baseless Info | This is information related to Bruce Wayne is not mentioned in the source content. | yes | yes |
| 4 | 2121 | test | summarization | Evident Baseless Info | *not mentioned* | yes | yes |
| 5 | 2243 | test | summarization | Evident Baseless Info | No mentioned of the author being a civil rights lawyer in the source content. | yes | yes |
| 6 | 2878 | test | summarization | Evident Baseless Info | *not mentioned* | yes | yes |
| 7 | 2937 | calib | summarization | Evident Baseless Info | This info is not mentioned in the source content. | yes | yes |
| 8 | 3082 | test | summarization | Evident Conflict | "After weeks... in the Boston Marathon bombing trial." (The exact number of weeks is not mentioned.) | no | no |
| 9 | 3146 | test | summarization | Evident Baseless Info | The specific date was not mentioned | yes | yes |
| 10 | 3149 | test | summarization | Evident Baseless Info | It is not mentioned that that police identified her through her IP address. | yes | yes |
| 11 | 3861 | test | summarization | Evident Baseless Info | No mention of these details in the source content. | yes | yes |
| 12 | 3873 | test | summarization | Evident Baseless Info | No mention of such study | yes | yes |
| 13 | 4442 | test | summarization | Evident Baseless Info | No mention of the exact number of years that Wahl have kept the position. | yes | yes |
| 14 | 5909 | calib | data2text | Subtle Baseless Info | *not mentioned but could be part "Wine Bar" catergory* | yes | no |
| 15 | 6767 | test | data2text | Subtle Baseless Info | Indeed, Korean-style barbecue is a Korean cuisine, however, it is not mentioned in the data. | yes | yes |
| 16 | 6772 | test | data2text | Evident Baseless Info | No mention of the atmosphere being cozy. | yes | yes |
| 17 | 6920 | test | data2text | Evident Baseless Info | This reviewer complained that his/her food was so salty and so spicy. No mention that it was delivered late. | yes | yes |
| 18 | 7036 | test | data2text | Evident Baseless Info | This information is not mentioned in the business info or reviews. | yes | yes |
| 19 | 7204 | test | data2text | Evident Baseless Info | No mention of delivery options and online orders | yes | yes |
| 20 | 7294 | test | data2text | Evident Baseless Info | No mentioned of this info. | yes | yes |
| 21 | 7328 | calib | data2text | Evident Baseless Info | No mentioned of vegetarian and gluten-free options | yes | yes |
| 22 | 7821 | calib | data2text | Evident Baseless Info | It is not mentioned in the original source that Bougainvillea is a locally owned restaurant. | yes | yes |
| 23 | 7919 | test | data2text | Evident Baseless Info | *not mentioned* | yes | yes |
| 24 | 8108 | calib | data2text | Evident Baseless Info | "lot": true (but not stated whether it is free or not) | no | no |
| 25 | 9081 | test | data2text | Evident Baseless Info | *not mentioned* | yes | yes |
| 26 | 9248 | calib | data2text | Evident Baseless Info | It is not mentioned if Jeff was the owner. | yes | yes |
| 27 | 9770 | test | data2text | Evident Baseless Info | Not mentioned in the structured data. | yes | yes |
| 28 | 9771 | test | data2text | Evident Baseless Info | This specific info is not mentioned in any of the reviews. | yes | yes |
| 29 | 9832 | calib | data2text | Subtle Baseless Info | No mention of other menus related to hotdogs | yes | yes |
| 30 | 11446 | test | data2text | Evident Baseless Info | *not mentioned* | yes | yes |
| 31 | 11521 | test | data2text | Evident Baseless Info | *not mentioned* | yes | yes |
| 32 | 12260 | test | qa | Evident Baseless Info | These steps are not mentioned in the passage 3 (steps in setting up Mozilla Firefox as default browser.) | yes | yes |
| 33 | 12449 | test | qa | Evident Baseless Info | *this is not mentioned in the sourced data* | yes | yes |
| 34 | 13094 | calib | qa | Subtle Baseless Info | *not mentioned in the sourced data* | yes | yes |
| 35 | 13363 | test | qa | Subtle Baseless Info | *not mentioned* | yes | yes |
| 36 | 13376 | test | qa | Evident Baseless Info | *this is not mentioned* | yes | yes |
| 37 | 13865 | calib | qa | Evident Baseless Info | No mention of the color white representing innocence | yes | yes |
| 38 | 14133 | test | qa | Evident Baseless Info | *this is not mentioned in Passage 3* | yes | yes |
| 39 | 14133 | test | qa | Evident Baseless Info | *this is not mentioned in Passage 2* | yes | yes |
| 40 | 14146 | calib | qa | Subtle Baseless Info | *did not mentioned these as "two compartments"* | yes | yes |
| 41 | 14572 | calib | qa | Subtle Baseless Info | It is true that the symptoms of Raynaud's disease appear due to cold temperature, however, this is not mentioned in the passages. | yes | yes |
| 42 | 14715 | test | qa | Evident Baseless Info | No mention of the items being safe to consume for those people with gluten sensitivity or celiac disease | yes | yes |
| 43 | 14806 | test | qa | Evident Baseless Info | No mention of the current temperature in Bucharest | yes | yes |
| 44 | 14942 | calib | qa | Evident Baseless Info | *not mentioned* | yes | yes |
| 45 | 15178 | test | qa | Evident Conflict | This information is found in passage 2, not in passage 1. | no | no |
| 46 | 15836 | calib | qa | Subtle Baseless Info | *not mentioned* | yes | yes |
| 47 | 16053 | calib | qa | Evident Baseless Info | *not mentioned in the source data* | yes | yes |
| 48 | 16060 | test | qa | Evident Baseless Info | These details are not mentioned in the passages. | yes | yes |
| 49 | 16898 | test | qa | Subtle Baseless Info | *this is a fact but not stated in the sourced data* | yes | yes |
| 50 | 17621 | calib | qa | Evident Baseless Info | This particular detail was not mentioned in any of the passages | yes | yes |
