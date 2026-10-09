partial: re-test on one public serum miRNA cohort (GEO GSE137140): the abstract's pattern (medium-dysregulation miRNAs classify lung cancer at ~99%) is reproduced, but class is perfectly confounded with sample cohort and 50 random miRNAs do almost as well, so it is NOT shown to be lung-cancer biology; the K-means n=31 is not reproduced. Student's own data not available.
# doc230-cbio033-mirna-lung
10 directions (D1-D10), PREREG.md + src/run033.py committed together (244766b) before any model was scored. Results and wording: RESULTS.md. Raw: results/.
Reproduce: download GSE137140_series_matrix.txt.gz, GPL21263 SOFT text and RNAcentral mirbase.fasta.gz into data/ (hashes in data_manifest.json), then `python3 src/run033.py` (about 12 minutes on 2 CPU, 2 GB). Python 3.10, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2.
Data and per-direction numbers are in results/results033.json. Constitution: real public data, CPU only, no stubs.
