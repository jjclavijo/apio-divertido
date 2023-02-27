#!/bin/bash

tmpdir=$(mktemp -d -p .)

cd $tmpdir

python -m eletor.simulatenoise -i ../valid_ctls/sn1.ctl
ls -lh dir

python -m eletor.simulatenoise -i ../valid_ctls/sn2.ctl <<EOF
0.1
-0.7
0.2
EOF
ls -lh obs_files

cd ..

rm -r $tmpdir 
