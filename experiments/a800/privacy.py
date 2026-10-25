"""Current chapter 8 implementation; FaceNet historical baseline is privacy_facenet_baseline.py."""
from pathlib import Path
import runpy

def main():
    here = Path(__file__).resolve()
    candidates = [p / rel for p in here.parents for rel in
                  ('examples/minimal/08_privacy.py', 'practice/minimal/08_privacy.py')]
    target = next((p for p in candidates if p.is_file()), None)
    if target is None:
        raise SystemExit('Use the complete teaching repository or accompanying book project.')
    runpy.run_path(str(target), run_name='__main__')

if __name__ == '__main__':
    main()
