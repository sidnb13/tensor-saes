#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

bash scripts/pythia/pythia14m_cross_rp1t.sh
bash scripts/pythia/pythia31m_cross_rp1t.sh
bash scripts/pythia/pythia70m_cross_rp1t.sh