w = 4.0
w2 = 2.0
lr = 10e-3

i = 2.0
t = 10.0

for _ in range(10000):
    o = i * w
    e = t - o
    d = w
    w += e * lr * d
    print(f"w: {w}, o: {o}, e: {e}")
print(f"w: {w}, o: {o}, e: {e}")


# sidenote figure out derivative brosky
# + 1
# * w
# / -w