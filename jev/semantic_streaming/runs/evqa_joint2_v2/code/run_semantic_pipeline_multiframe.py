#!/usr/bin/env python3
"""Joint-ROI entry point reusing the frozen v1 encoder, control and stage engine."""
import run_semantic_pipeline as engine
from pipeline_roi_multiframe import locate

CODE_FILES = (*engine.CODE_FILES, 'pipeline_roi_multiframe.py', 'run_semantic_pipeline_multiframe.py')


def main():
    engine.locate = locate
    engine.CODE_FILES = CODE_FILES
    return engine.main()


if __name__ == '__main__':
    raise SystemExit(main())
