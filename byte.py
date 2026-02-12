python -c "p=3808; b=open('median.py','rb').read(); print('len',len(b)); print('byte@p',hex(b[p])); print('window',b[p-20:p+20])"
