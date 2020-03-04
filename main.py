import pandas as pd
import numpy as np


def readcsv(filename):
    data = pd.read_csv(filename, usecols=[i for i in range(6)])
    return np.array(data)


def strtottime(st):
    qtime = st.split(":")
    mininseconds = float(qtime[0]) * 60
    return mininseconds + float(qtime[1]) + float(qtime[2]) / 1000

# for row in regions.itertuples():
#     print(row.Sem)
regions = pd.read_csv("regions.csv")
first = pd.read_csv("question_time.csv")
#first = readcsv("User 1_all_gaze.csv")
#first = pd.read_csv("User 1_all_gaze.csv", usecols=[i for i in range(6)])

for index, row in first.iterrows():
    print(row[0])
    current_file = pd.read_csv("sotis/data/user{0}/User {0}_all_gaze.csv".format(row[0]), usecols=[i for i in range(6)])
    rdata = []
    sp = None
    seq = 1
    frist_question_start = strtottime(row[1])
    timeperquestion = []
    previoustime = 0

    avg_time = 0

    for datarow in current_file.itertuples():
        if datarow.FPOGV == 0:
            continue
        else:
            x = datarow.FPOGX * 1920
            y = datarow.FPOGY * 1080
            s = -1
            for reg_row in regions.itertuples():
                if (reg_row.X <= x <= (reg_row.X + reg_row.Xlen)) and (reg_row.Y <= y <= (reg_row.Y + reg_row.Ylen)):
                    s = reg_row.Sem
                    # print("rowX:{0},X:{1}, rowXlen:{2}, rowY:{3},Y:{4}, rowYlen:{5}, S:{6}".format(reg_row.X, x, reg_row.X + reg_row.Xlen, reg_row.Y,
                    #                                                                       y, reg_row.Y + reg_row.Ylen, s))
            if s == -1:
                s = 6 # else..
            question_time = strtottime(row[seq])

            if datarow.TIME > question_time:
                seq = seq + 1
            if seq == 34:
                break
            deq = (seq, datarow.TIME, datarow.TIMETICK, datarow.FPOGX, datarow.FPOGY, datarow.FPOGD, datarow.FPOGV,s)

            # grouping same gazes
            if sp is not None:
                if sp[7] != s:
                    rdata.append(sp)
                    sp = None
                else:
                    # 0     1       2       3      4      5      6    7
                    (seq, time, timetick, fpogx, fpogy, fpogd, fpogv, s) = sp
                    q1 = round((datarow.FPOGX + fpogx)/2, 5)
                    q2 = round((datarow.FPOGY + fpogy)/2, 5)
                    sp = (seq, datarow.TIME, datarow.TIMETICK, q1, q2, datarow.FPOGD + fpogd, datarow.FPOGV, s)
            else:
                sp = deq

            #rdata.append(deq) #u slucaju svih podataka

    ofile = open("output/User {0}_output_g.csv".format(row[0]), "w")
    ofile.write("Seq, fpogx, fpogy, fpogd, s\n")
    print("Writing to file {0}".format(row[0]))
    # u slucaju grupisanja
    pan = pd.DataFrame(rdata, columns=['Seq','Time', 'TimeTick', 'Fpogx', 'Fpogy', 'Fpogd', 'Fpogv', 'Region'])
    pan = pan.sort_values(by=['TimeTick'], ascending=True)
    a = 0
    i = 0
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    #print(pan.describe())
    #avg_time = pan['Fpogd'].mean()
    bin_labels = ['0', '1', '2'] #['brzo', 'prosecno', 'sporo'] #
    # pan['Fpogd2'] = pd.qcut(pan['Fpogd'], 3,
    #                               #q=[0, .2, .4, .6, .8, 1],
    #                               labels=bin_labels)

    pan['Fpogd'] = pd.qcut(pan['Fpogd'], 3, labels=bin_labels)

    #print(pan)

    for seq, time, timetick, fpogx, fpogy, fpogd, fpogv, s in pan.values.tolist():#rdata:  #pan.values.tolist(): # u slucaju grupisanja
        ofile.write("{0},{2},{3},{4},{5}\n".format(seq, round(time, 5), round(abs(fpogx), 5), round(abs(fpogy), 5), int(fpogd), s))

