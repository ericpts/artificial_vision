#!/usr/bin/env python3

import re
from pathlib import Path

import matplotlib.pyplot as plt

def main():
    results = []
    for resf in Path('results').iterdir():
        with resf.open('r+t') as f:
            score = float(f.read())
        m = re.match(r'(\w+)_(\d+)\.txt', resf.name)
        (csf, nclusters) = (m[1], m[2])

        results.append((csf, int(nclusters), score))

    classifiers = list(set(map(lambda r: r[0], results)))

    for csf in classifiers:
        points = list(map(lambda r: (r[1], r[2]), filter(lambda r: r[0] == csf, results)))
        (ncluster_vector, score_vector) = list(zip(*sorted(points)))

        print(ncluster_vector, score_vector)
        plt.xscale('log')
        plt.plot(ncluster_vector, score_vector, 'b-')
        plt.plot(ncluster_vector, score_vector, 'go')
        plt.xlabel('Number of clusters')
        plt.ylabel('Accuracy')
        plt.savefig('csf_{}.svg'.format(csf))
        plt.close()


if __name__ == '__main__':
    main()
