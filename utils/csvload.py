import csv
def cut_ind(x, indices):
    rv = x.copy()
    h = np.arange(len(indices))
    indices = np.array(indices)
    indices -= h
    for i in indices:
        tmp = rv[:, i+1:]
        rv = np.hstack( (rv[:, :i], tmp) )
    return rv


def lcsv(fnm, delim = ',', quote = '"', blank_str = 'Blank', qc_str = 'QC', sample_str = 'Sample'):
# here we open file and work with it
    with open(fnm) as csvfile:
        # these csv files have multi-header; here we skip the level-one
        hd1 = csvfile.readline()
        # and here we skip the level-two. Level-3 will be parsed automatically
        hd2 = csvfile.readline()
        hd2 = hd2.replace('"', '')
        sph2 = hd2.split(delim)
        hd2_m = [blank_str, qc_str, sample_str]
        offset = []
        for s in hd2_m:
            if s is None:
                continue
            offset.append(sph2.index(s))
        # read file as a 
        reader = csv.DictReader(csvfile, delimiter=delim, quotechar=quote, lineterminator = '\n')
        fieldnames = reader.fieldnames

        mz = []
        rt = []
        blank = []
        qc = []
        sample = []
        idx = 1

        for row in reader:
            mz.append(float(row['m/z']))
            rt.append(float(row['Retention time (min)']))
            blank_row = []
            for fni in fieldnames[offset[0] : offset[1]]:
                blank_row.append( float(row[fni]) )
            blank.append(blank_row)
            if qc_str is not None:
                qc_row = []
                for fni in fieldnames[offset[1] : offset[2]]:
                    qc_row.append( float(row[fni]) )
                idx += 1
                qc.append(qc_row)
            sample_row = []
            for fni in fieldnames[offset[idx] : ]:
                sample_row.append( float(row[fni]) )
            sample.append(sample_row)
    return mz, rt, blank, qc, sample

def loadpar(fnm):
    with open(fnm) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"', lineterminator = '\n')
        fieldnames = reader.fieldnames
        # 1 and 2 = Vial and Sample name
        trial_ind = 1
        sname_ind = 2
        nrun_ind  = 0
        trial = []
        blank = []
        flav  = []
        for row in reader:
            rec = row[fieldnames[sname_ind]]
            numtrial =  int(row[fieldnames[trial_ind]])
            run = int( row[fieldnames[nrun_ind]] )
            if rec.startswith("Flav"):
                flav.append([run, numtrial, rec])
            elif rec.startswith("Blank"):
                blank.append([run, numtrial, rec])
            elif rec[0] in '1234567890':
                trial.append([run, numtrial, rec])
            else:
                print "loadpar: Unexpected!", rec
            
    return blank, flav, trial


# from this point file is automatically closed. All work with dictionary like
# setting variables must be done in "with"-block

# Print all fields in dictionary. Note - each row is a new dictionary, and can't be reached after closing file.

if __name__ == '__main__':
    # filename
    fnm = '../data/peak_tables/peak_table_Neg.csv'

    mz, rt, b, q, sample = lcsv(fnm)


    #print reader.fieldnames

    # Blank -> intensity matrix
