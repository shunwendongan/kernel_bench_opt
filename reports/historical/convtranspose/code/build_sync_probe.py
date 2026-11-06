from pathlib import Path
import json, subprocess, shutil

p = Path('${BOARD_PROJECT}/experiments/convtranspose_opt_20261105')
(p / 'src/sync_probe').mkdir(exist_ok=True)
a = json.loads((p / 'records/compile_omni.cpp.json').read_text())
a[a.index('-c') + 1] = str(p / 'src/omni.sync_probe.cpp')
a[a.index('-o') + 1] = str(p / 'src/sync_probe/omni.cpp.o')
(p / 'records/compile_sync_probe.json').write_text(json.dumps(a, indent=2))
subprocess.run(a, check=True)
shutil.copy2(p / 'bin/libMNN_candidate.a', p / 'bin/libMNN_sync_probe.a')
subprocess.run(['ar', 'r', str(p / 'bin/libMNN_sync_probe.a'), str(p / 'src/sync_probe/omni.cpp.o')], check=True)
a = json.loads((p / 'records/link_omni_ab.json').read_text())
a = [x.replace('libMNN_candidate.a', 'libMNN_sync_probe.a') for x in a]
a[a.index('-o') + 1] = str(p / 'bin/omni_ab_sync')
(p / 'records/link_omni_ab_sync.json').write_text(json.dumps(a, indent=2))
subprocess.run(a, check=True)
print('SYNC_BUILD_OK', flush=True)
