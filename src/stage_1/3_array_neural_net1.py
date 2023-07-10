weights = (1.0, 2.0, 3.0, 4.0)
lr = 10e-3

inputs = (1.0, 2.0, 3.0, 4.0)
target = (8.0, 2.0, 3.0, 6.0)

for _ in range(1000):
    # print(weights)
    # print(target)
    output = list((ii * ww for ii, ww in zip(inputs, weights)))
    # print(output)
    error = list((t - o for t, o in zip(target, output)))
    # print(error)
    gradient = weights
    weights = list((ww + lr * ee * gg for ww, ee, gg in zip(weights, error, gradient)))
    # print(weights)
    print(f"e: {error}, o: {output}, w: {weights}")
