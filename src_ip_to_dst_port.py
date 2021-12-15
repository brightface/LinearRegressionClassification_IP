import pandas as pd

df = pd.read_csv('04_hashed.csv')
#df

# 建立字典
src_ip = []
dst_port = []
dict_src_ip = {}
dict_dst_ip = {}
# 标签src_ip
for i in range(len(df)):
  print('ip == ',len(src_ip))
  if df['src_ip'][i] not in src_ip:
    src_ip.append(df['src_ip'][i])

#标签dst_port
for i in range(len(df)):
  print('dst_port == ',len(dst_port))
  if df['dst_port'][i] not in dst_port:
    dst_port.append(df['dst_port'][i])

with open('src_ip_to_dst.txt', 'w', encoding="utf-8") as f:

    for i in range(len(df)):
        x1 = str((src_ip.index(df['src_ip'][i])+1))
        x2 = str((dst_port.index(df['dst_port'][i])+1))
        action = str(df['Action'][i])
        data = x1 +' ' + x2 + ' ' + action +'\n'
        f.write(data)