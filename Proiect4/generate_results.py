#!/usr/bin/env python3

import os
import subprocess

import concurrent.futures
from pathlib import Path
from typing import List, Tuple

from yaml import *


def calculate_score(conf_path: Path) -> float:
    return float(subprocess.check_output(['python3', 'main.py', '-c', str(conf_path)]))


def generate_scenarios() -> List[Tuple[Path, str, int]]:
    """ Returns list of (conf_path, classifier, nclusters). """
    ret = []
    with open('conf.yml', 'r') as f:
        data = load(f)

        for nclusters in [5, 10, 25, 50, 100, 200, 500, 1000]:
            data['clusters'] = nclusters
            for csf in ['SVM', 'NN']:
                conf_dir = Path('.') / 'confs' / csf
                os.makedirs(str(conf_dir), exist_ok=True)
                conf_path = conf_dir / 'conf_{}.yml'.format(nclusters)

                data['classifier'] = csf
                with conf_path.open('w+t') as g:
                    g.write(dump(data, default_flow_style=False))

                ret.append((conf_path, csf, nclusters))
    return ret


def main():
    scenarios = generate_scenarios()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(calculate_score, conf) for (conf, _, _) in scenarios]

    scores = {(csf, nclusters): future.result()
              for ((_, csf, nclusters), future) in zip(scenarios, futures)}

    results_dir = Path('./results')
    os.makedirs(str(results_dir), exist_ok=True)
    print(scores)

    for ((csf, nclusters), score) in scores.items():
        with (results_dir / '{}_{}.txt'.format(csf, nclusters)).open('w+t') as f:
            f.write(str(score) + "\n")


if __name__ == '__main__':
    main()
