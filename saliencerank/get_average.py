f = open("result_raw_salience.txt")
count = 1
f1 = 0
for line in f:
    if count%3 == 0:
        line = line.split()
        print(float(line[1]))
        f1 += float(line[1])
    count += 1
print(f1/1000)