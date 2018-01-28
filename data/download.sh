#!/usr/bin/env bash
for f in hhold indiv; do for a in A B C; do for t in train test; do wget https://s3.amazonaws.com/drivendata/data/50/public/"$a"_"$f"_"$t".csv; done; done; done
