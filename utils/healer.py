import os
from mmap import mmap

def comp_offset(fname):
	f = open(fname, 'r+')
	offset = 0
	for i in xrange(3):
		a = f.readline()
		offset += len(a)
	a = f.readline()
	a = a.replace("></parentFile>", '')# ' fileSha1="" ></parentFile>')
	offset += len(a)
	f.close()
	return offset - 2


def insert(filename, str, pos):
    if len(str) < 1:
        # nothing to insert
        return

    f = open(filename, 'r+')
    m = mmap(f.fileno(), os.path.getsize(filename))
    origSize = m.size()

    # or this could be an error
    if pos > origSize:
        pos = origSize
    elif pos < 0:
        pos = 0

    m.resize(origSize + len(str))
    m[pos+len(str):] = m[pos:origSize]
    m[pos:pos+len(str)] = str
    m.close()
    f.close()

if __name__ == "__main__":
	dirname = './data/'
	fnms = os.listdir(dirname)
	for fn in fnms:
		dfn = dirname + fn
		offset = comp_offset(dfn)
        insert(dfn, ' fileSha1="" ', offset)
        
