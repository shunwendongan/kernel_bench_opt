from pathlib import Path
import datetime, json, os, subprocess, sys, time

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
args = sys.argv[2:]
paths = list(Path('/sys/class/thermal').glob('thermal_zone*/temp')) + list(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')) + list(Path('/sys/class/devfreq').glob('*/cur_freq'))
selected = {k:v for k,v in os.environ.items() if k.startswith(('OMNI_', 'DECONV_','BIGVGAN_')) or k == 'LD_LIBRARY_PATH'}
(out/'command.json').write_text(json.dumps({'argv':args,'environment':selected,'cwd':str(Path.cwd()),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
with (out/'stdout.log').open('w') as log, (out/'telemetry.jsonl').open('w') as telemetry:
    start = time.monotonic()
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    while p.poll() is None:
        values={}
        for path in paths:
            try: values[str(path)]=int(path.read_text().strip())
            except (OSError,ValueError): pass
        telemetry.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_s':time.monotonic()-start,'pid':p.pid,'loadavg':Path('/proc/loadavg').read_text().strip(),'values':values})+'\n')
        telemetry.flush()
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired: pass
    (out/'exit.json').write_text(json.dumps({'exit_code':p.returncode,'wall_s':time.monotonic()-start},indent=2))
sys.exit(p.returncode)
