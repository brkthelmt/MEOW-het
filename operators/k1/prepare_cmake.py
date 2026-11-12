"""Copy CANN build helpers locally; normalize Ninja object paths in the copy."""
from pathlib import Path
import shutil
import sys
src=Path(sys.argv[1]);dst=Path(sys.argv[2])
shutil.copytree(src,dst,dirs_exist_ok=True)
p=dst/'legacy_modules/util/extract_host_stub.py'
s=p.read_text()
old='source_mapping[func_group.filepath]'
assert old in s, 'Recheck helper compatibility for this CANN version'
s=s.replace(old,'source_mapping[os.path.abspath(func_group.filepath)]')
s=s.replace('absolute_obj = os.path.join(directory, output_file)',
            'absolute_obj = os.path.abspath(os.path.join(directory, output_file))')
p.write_text(s)
print('Prepared isolated CANN helper copy:',dst)
