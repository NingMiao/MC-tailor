import sys

f=open(sys.argv[1])
g=open(sys.argv[2],'w')
l=int(sys.argv[3])

for i in range(l):
  g.write(f.readline())

g.close()
