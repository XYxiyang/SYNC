# Metric aggregation

`scripts/summarize_metric.py` accepts CSV files with a `smiles_score` column and
reports the number of input, valid and failed records, the mean score, ES fraction,
and labeled accuracy. Nonnumeric and nonfinite values are counted as failed.

The classification thresholds follow Appendix C:

| Metric argument | ES decision | Representation |
|---|---|---|
| `sa` | score > 0.61 | `(10 - raw_SA) / 9`, rounded to two decimals by the SA driver |
| `syba` | score > -18.6 | SYBA score |
| `fs` | score > -13 | FSScore |
| `ra` | score > 0.5 | RAscore |
| `sc` | score < 3.1 | SCScore |

```bash
python scripts/summarize_metric.py --metric sa \
  --positive data/sa/easy.csv --negative data/sa/hard.csv \
  --output outputs/sa_accuracy.json
python scripts/summarize_metric.py --metric sc \
  --unlabeled data/sc/generated.csv --output outputs/sc_generated.json
```

The default `--accuracy-rule legacy` follows the original evaluation script:
for higher-is-better metrics, ES must be strictly above and HS strictly below
the threshold. For SCScore, the script complements that count.
`--accuracy-rule binary` uses the ES decision in the table and assigns equality
to HS. Results include the rule and the number of threshold-equal samples.
`es_fraction` always follows the ES decision in the table.

Metric drivers and AiZynthFinder configuration examples are in
`legacy/benchmark/`. Install the relevant metric packages, pretrained models,
reaction templates and stock files, then configure their dataset paths.
Keep molecule identities and the score scale when exporting metric CSV files.
