# fix_encoding.py

import pathlib

FILE_TO_FIX = "median.py"

p = pathlib.Path(FILE_TO_FIX)
b = p.read_bytes()

bad_count = b.count(b'\xA0')
print("Found", bad_count, "non-breaking spaces (0xA0)")

if bad_count > 0:
    b = b.replace(b'\xA0', b' ')
    p.write_bytes(b)
    print("Replaced 0xA0 with normal spaces.")
else:
    print("No 0xA0 found.")

print("Done.")
