b=open('median.py','rb').read(); print('len=',len(b)); print('n_0xA0=', b.count(b'\xA0')); print('first_offsets=', [i for i,ch in enumerate(b) if ch==0xA0][:20])
