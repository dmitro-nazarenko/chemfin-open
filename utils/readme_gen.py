import csv

fnm = ['neg.csv', 'pos.csv']
out = 'out.txt'

l = []
for f in fnm:
    with open(f, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='"')
        i = 0
        for row in reader:
            if i == 2:
                l.append(row)
                break
            i += 1

t = []
for i in xrange(len(fnm)):
    h = [x.replace('"', '') for x in l[i]]
    t.append(h)

f = open(out, 'a+')

l1 = len(t[0])
l2 = len(t[1])
for i in xrange(max(l1,l2)):
    line = ''
    if i < l1:
        line += t[0][i]
    else:
        line += '(none)'
    line += '           '
    if i < l2:
        line += t[1][i]
    else:
        line += '(none)'
    line += '\n'
    f.writelines(line)

f.close()
