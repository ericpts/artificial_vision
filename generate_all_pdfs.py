#!/usr/bin/python3

import sys
import subprocess
import concurrent.futures

from pathlib import Path


def generate_pdf_for_project(projectn: int, out: Path):
    print('Generating pdf for project {}'.format(projectn))
    subprocess.check_call(
        ['python3', 'generate_pdf.py', str(out)],
        cwd=Path('Proiect{}'.format(projectn)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def main():
    executor = concurrent.futures.ProcessPoolExecutor()
    for i in range(1, 4):
        out = Path('project_{}.pdf'.format(i))
        executor.submit(generate_pdf_for_project, i, out.resolve())
    executor.shutdown()


if __name__ == "__main__":
    main()
