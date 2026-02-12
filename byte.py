b=open('median.py','rb').read(); \
import codecs; \
try: b.decode('utf-8'); print('OK: valid utf-8'); \
except UnicodeDecodeError as e: print('BAD utf-8:', e, 'byte=', hex(b[e.start]))
