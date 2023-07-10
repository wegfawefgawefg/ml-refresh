w = 3.0
w2 = 1.0
lr = 10e-3

i = 2.0
t = 10.0

for _ in range(10000):
    o = i * w + w2
    
    e = t - o
    d_w = w
    d_w2 = 1
    
    w += e * lr * d_w
    w2 += e * lr * d_w2

    print(f"w: {w}, o: {o}, e: {e}")
print(f"w: {w}, w2: {w2}, o: {o}, e: {e}")


# sidenote figure out derivative brosky
# + 1
# * w
# / -w