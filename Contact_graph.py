import matplotlib.pyplot as plt

f = open('Contact.txt', 'r')
print(f)
results = []
y = []
x = []
i = 0

for line in f:
    i += 1
    line = line.split(',')
    results.append([i, int(line[1])])


for i in results:
    x.append(i[0])
    y.append(i[1])
f.close()

p = plt.plot(x, y, c = "g")
plt.xlabel('ms')
plt.ylabel('VPG')
plt.show()
