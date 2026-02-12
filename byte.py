b=open('median.py','rb').read(); print('n_0xA0=', b.count(b'\\xA0')); print('first_5_offsets=', [i for i in range(len(b)) if b[i]==0xA0][:5])
